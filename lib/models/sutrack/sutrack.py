"""SUTrack + research-grounded SLA-v3.

SLA-v3 keeps the native Fast-iTPN/CLIP/CENTER stack and adds three progressive
modules: ISIA -> LSC -> SGCR. Each structural intervention is residual and starts
from the native/SLA-v1 behavior.
"""
import torch
import math
from torch import nn

from .encoder import build_encoder
from .clip import build_textencoder
from .decoder import build_decoder
from .task_decoder import build_task_decoder
from .sla_v3 import GroundedSLA
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SUTRACK(nn.Module):
    """SUTrack with optional SLA-v3 modules."""

    def __init__(self, text_encoder, encoder, decoder, task_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average", cfg=None):
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.task_decoder = task_decoder
        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template

        self.sla_v3_enabled = False
        self.sla_v3 = None
        if cfg is not None and hasattr(cfg.MODEL, 'SLA_V3'):
            self.sla_v3_enabled = bool(getattr(cfg.MODEL.SLA_V3, 'ENABLED', False))
            if self.sla_v3_enabled:
                if decoder_type != 'CENTER':
                    raise ValueError('SLA-v3 currently requires MODEL.DECODER.TYPE=CENTER.')
                if not bool(getattr(cfg.DATA, 'MULTI_MODAL_LANGUAGE', False)):
                    raise ValueError('SLA-v3 requires DATA.MULTI_MODAL_LANGUAGE=True.')
                self.sla_v3 = GroundedSLA(int(self.encoder.num_channels), cfg.MODEL.SLA_V3)

    def forward(self, text_data=None,
                template_list=None, search_list=None, template_anno_list=None,
                text_src=None, task_index=None,
                feature=None, mode="encoder"):
        if mode == "text":
            return self.forward_textencoder(text_data)
        elif mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
        elif mode == "decoder":
            return (self.forward_decoder(feature, template_anno_list=template_anno_list),
                    self.forward_task_decoder(feature))
        else:
            raise ValueError(f'Unknown forward mode: {mode}')

    def forward_textencoder(self, text_data):
        return self.text_encoder(text_data)

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        return self.encoder(template_list, search_list, template_anno_list, text_src, task_index)

    def _token_ranges(self, tokens):
        start = 1 if self.class_token else 0
        ns = self.num_patch_x * self.num_frames
        nz = self.num_patch_z
        search_slice = slice(start, start + ns)
        template_start = start + ns
        static_slice = slice(template_start, template_start + nz)
        return search_slice, static_slice

    @staticmethod
    def _normalize_template_bbox(template_anno_list):
        if template_anno_list is None or len(template_anno_list) < 1:
            raise RuntimeError('SLA-v3 requires the static-template annotation.')
        bbox = template_anno_list[0]
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)
        if bbox.dim() > 2:
            bbox = bbox.view(bbox.size(0), -1, 4)[:, 0, :]
        else:
            bbox = bbox.view(bbox.size(0), 4)
        return bbox

    def _sla_context(self, tokens, template_anno_list):
        if not self.sla_v3_enabled:
            return None
        search_slice, static_slice = self._token_ranges(tokens)
        if tokens.size(1) <= static_slice.stop:
            raise RuntimeError(
                'SLA-v3 expected a fused language token after search/template tokens. '
                'Check DATA.MULTI_MODAL_LANGUAGE and USE_NLP.'
            )
        search_tokens = tokens[:, search_slice, :]
        static_tokens = tokens[:, static_slice, :]
        text_token = tokens[:, -1, :]
        static_bbox = self._normalize_template_bbox(template_anno_list)
        ctx = self.sla_v3.context(search_tokens, static_tokens, text_token, static_bbox)
        ctx['native_search_tokens'] = search_tokens
        ctx['raw_text_token'] = text_token
        return ctx

    def forward_decoder(self, feature, gt_score_map=None, template_anno_list=None):
        tokens = feature[0]
        search_slice, _ = self._token_ranges(tokens)
        native_search_tokens = tokens[:, search_slice, :]

        sla_ctx = self._sla_context(tokens, template_anno_list) if self.sla_v3_enabled else None
        search_tokens = sla_ctx['calibrated_search'] if sla_ctx is not None else native_search_tokens

        bs, _, c = search_tokens.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            dec_feature = search_tokens.permute((0, 2, 1)).contiguous()
            dec_feature = dec_feature.view(bs, c, self.fx_sz, self.fx_sz)
        else:
            dec_feature = search_tokens

        if self.decoder_type == "CORNER":
            pred_box, score_map = self.decoder(dec_feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            return {'pred_boxes': outputs_coord.view(bs, 1, 4), 'score_map': score_map}

        if self.decoder_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.decoder(dec_feature, gt_score_map)
            native_bbox = bbox.view(bs, 1, 4)
            out = {
                'pred_boxes': native_bbox,
                'native_pred_boxes': native_bbox,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map,
            }

            if sla_ctx is not None:
                resolved = self.sla_v3.resolve(
                    score_map_ctr,
                    size_map,
                    offset_map,
                    sla_ctx['native_search_tokens'],
                    sla_ctx['identity_anchor'],
                )
                # The actor uses native_pred_boxes for the original SUTrack box loss;
                # pred_boxes reflects the actual SLA-v3 inference path.
                out['pred_boxes'] = resolved['final_boxes'].unsqueeze(1)
                out['sla_native_search_tokens'] = sla_ctx['native_search_tokens']
                out['sla_calibrated_search_tokens'] = sla_ctx['calibrated_search']
                out['sla_raw_text_token'] = sla_ctx['raw_text_token']
                out['sla_identity_anchor'] = sla_ctx['identity_anchor']
                out['sla_target_summary'] = sla_ctx['target_summary']
                out['sla_context_summary'] = sla_ctx['context_summary']
                out['sla_target_coverage'] = sla_ctx['target_coverage']
                out['sla_target_context_attn'] = sla_ctx['target_context_attn']
                out['sla_identity_res_scale'] = sla_ctx['identity_res_scale']
                out['sla_semantic_field'] = sla_ctx['semantic_field']
                out['sla_identity_logits'] = sla_ctx['identity_logits']
                out['sla_spatial_gate'] = sla_ctx['spatial_gate']
                for key, value in resolved.items():
                    out['sgcr_' + key] = value
            return out

        if self.decoder_type == "MLP":
            score_map, bbox, offset_map = self.decoder(dec_feature, gt_score_map)
            return {
                'pred_boxes': bbox.view(bs, 1, 4),
                'score_map': score_map,
                'offset_map': offset_map,
            }
        raise NotImplementedError

    def refine_candidates(self, response_map, out_dict):
        """Run SGCR with the actual test-time response (after Hann window)."""
        if not self.sla_v3_enabled:
            raise RuntimeError('refine_candidates called while SLA-v3 is disabled.')
        required = [
            'size_map', 'offset_map',
            'sla_native_search_tokens', 'sla_identity_anchor'
        ]
        missing = [k for k in required if k not in out_dict]
        if missing:
            raise RuntimeError(f'SLA-v3 refinement missing decoder fields: {missing}')
        return self.sla_v3.resolve(
            response_map,
            out_dict['size_map'],
            out_dict['offset_map'],
            out_dict['sla_native_search_tokens'],
            out_dict['sla_identity_anchor'],
        )

    def forward_task_decoder(self, feature):
        feature = feature[0]
        if self.task_feature_type == 'class':
            feature = feature[:, 0:1]
        elif self.task_feature_type == 'text':
            feature = feature[:, -1:]
        elif self.task_feature_type == 'average':
            feature = feature.mean(1).unsqueeze(1)
        else:
            raise NotImplementedError('task_feature_type must be chosen from class, text, and average')
        return self.task_decoder(feature)


def build_sutrack(cfg):
    encoder = build_encoder(cfg)
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        text_encoder = build_textencoder(cfg, encoder)
    else:
        text_encoder = None
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    model = SUTRACK(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        num_frames=cfg.DATA.SEARCH.NUMBER,
        num_template=cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE,
        cfg=cfg,
    )
    return model

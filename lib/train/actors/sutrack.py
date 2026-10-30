from . import BaseActor
from lib.utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_xywh_to_xyxy,
    box_xywh_to_cxcywh,
    box_iou,
    generalized_box_iou,
)
import math
import torch
import torch.nn.functional as F
from lib.utils.heapmap_utils import generate_heatmap


class SUTrack_Actor(BaseActor):
    """SUTrack actor with SLA-v1 supervision + research-grounded SLA-v3 modules."""

    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE

        # Proven SLA-v1 training objective; kept unchanged.
        self.sem_loc_weight = float(getattr(cfg.TRAIN, 'SEM_LOC_WEIGHT', 0.0))
        self.sem_loc_temperature = float(getattr(cfg.TRAIN, 'SEM_LOC_TEMPERATURE', 0.10))
        self.sem_loc_detach_text = bool(getattr(cfg.TRAIN, 'SEM_LOC_DETACH_TEXT', True))
        self.sem_loc_normalize_logn = bool(getattr(cfg.TRAIN, 'SEM_LOC_NORMALIZE_LOGN', True))

        # SLA-v3 objectives.
        self.identity_loc_weight = float(getattr(cfg.TRAIN, 'IDENTITY_LOC_WEIGHT', 0.0))
        self.anchor_contrast_weight = float(getattr(cfg.TRAIN, 'ANCHOR_CONTRAST_WEIGHT', 0.0))
        self.anchor_contrast_margin = float(getattr(cfg.TRAIN, 'ANCHOR_CONTRAST_MARGIN', 0.10))
        self.cand_cls_weight = float(getattr(cfg.TRAIN, 'CAND_CLS_WEIGHT', 0.0))
        self.cand_box_weight = float(getattr(cfg.TRAIN, 'CAND_BOX_WEIGHT', 0.0))
        self.cand_quality_temperature = float(getattr(cfg.TRAIN, 'CAND_QUALITY_TEMPERATURE', 0.20))
        self.cand_min_oracle_iou = float(getattr(cfg.TRAIN, 'CAND_MIN_ORACLE_IOU', 0.05))

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def _attach_sla_v1_semantic_localization(self, outputs, enc_opt):
        """Keep the exact SLA-v1 raw-text-to-search localization objective."""
        if (not self.multi_modal_language) or self.sem_loc_weight <= 0:
            return

        tokens = enc_opt[0]
        net_ref = self.net.module if hasattr(self.net, 'module') else self.net
        start = 1 if bool(getattr(net_ref, 'class_token', False)) else 0
        num_search = int(getattr(net_ref, 'num_patch_x')) * int(getattr(net_ref, 'num_frames', 1))
        if tokens.size(1) <= start + num_search:
            raise RuntimeError(
                'SLA expected a text token after search/template tokens. '
                'Check DATA.MULTI_MODAL_LANGUAGE and USE_NLP.'
            )

        search_tokens = tokens[:, start:start + num_search, :]
        text_token = tokens[:, -1, :]
        if self.sem_loc_detach_text:
            text_token = text_token.detach()

        search_tokens = F.normalize(search_tokens, dim=-1, eps=1e-6)
        text_token = F.normalize(text_token, dim=-1, eps=1e-6)
        logits = torch.sum(search_tokens * text_token.unsqueeze(1), dim=-1)
        logits = logits / max(self.sem_loc_temperature, 1e-6)
        outputs['sem_loc_logits'] = logits

    def forward_pass(self, data):
        b = data['search_images'].shape[1]
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b, dim=0)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b, dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b, dim=0)

        if self.multi_modal_language:
            text = data['nlp_ids'].permute(1, 0)
            text_src = self.net(text_data=text, mode='text')
        else:
            text_src = None

        task_index_batch = [self.cfg.MODEL.TASK_INDEX[key.upper()] for key in data['dataset']]
        task_index_batch = torch.tensor(task_index_batch, device=data['search_images'].device)

        enc_opt = self.net(
            template_list=template_list,
            search_list=search_list,
            template_anno_list=template_anno_list,
            text_src=text_src,
            task_index=task_index_batch,
            mode='encoder',
        )

        outputs, task_class_output = self.net(
            feature=enc_opt,
            template_anno_list=template_anno_list,
            mode='decoder',
        )
        task_class_output = task_class_output.view(-1, task_class_output.size(-1))
        outputs['task_class'] = task_class_output
        outputs['task_class_label'] = task_index_batch

        self._attach_sla_v1_semantic_localization(outputs, enc_opt)
        return outputs

    def _heatmap_ce(self, logits, gt_gaussian_maps):
        target = gt_gaussian_maps.flatten(1).detach().clamp_min(0.0)
        if target.size(1) != logits.size(1):
            raise RuntimeError(
                f'SLA grid mismatch: semantic tokens={logits.size(1)}, GT heatmap={target.size(1)}'
            )
        target_prob = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
        loss = -(target_prob * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        if self.sem_loc_normalize_logn:
            loss = loss / max(math.log(logits.size(1)), 1.0)

        with torch.no_grad():
            pred_idx = logits.argmax(dim=1)
            top5_idx = logits.topk(k=min(5, logits.size(1)), dim=1).indices
            max_t = target.max(dim=1, keepdim=True).values
            target_mask = target >= (0.25 * max_t)
            top1_hit = target_mask.gather(1, pred_idx.unsqueeze(1)).float().mean()
            top5_hit = target_mask.gather(1, top5_idx).any(dim=1).float().mean()
        return loss, top1_hit, top5_hit

    def _semantic_localization_loss(self, pred_dict, gt_gaussian_maps):
        if 'sem_loc_logits' not in pred_dict or self.sem_loc_weight <= 0:
            zero = gt_gaussian_maps.sum() * 0.0
            return zero, zero.detach(), zero.detach()
        return self._heatmap_ce(pred_dict['sem_loc_logits'], gt_gaussian_maps)

    def _identity_localization_loss(self, pred_dict, gt_gaussian_maps):
        if 'sla_identity_logits' not in pred_dict or self.identity_loc_weight <= 0:
            zero = gt_gaussian_maps.sum() * 0.0
            return zero, zero.detach(), zero.detach()
        return self._heatmap_ce(pred_dict['sla_identity_logits'], gt_gaussian_maps)

    def _anchor_contrast_loss(self, pred_dict):
        keys = ['sla_identity_anchor', 'sla_target_summary', 'sla_context_summary']
        if self.anchor_contrast_weight <= 0 or not all(k in pred_dict for k in keys):
            ref = pred_dict['score_map']
            return ref.sum() * 0.0
        q = F.normalize(pred_dict['sla_identity_anchor'], dim=-1, eps=1e-6)
        target = F.normalize(pred_dict['sla_target_summary'], dim=-1, eps=1e-6)
        context = F.normalize(pred_dict['sla_context_summary'], dim=-1, eps=1e-6)
        pos = torch.sum(q * target, dim=-1)
        neg = torch.sum(q * context, dim=-1)
        return F.relu(self.anchor_contrast_margin - pos + neg).mean()

    def _candidate_losses(self, pred_dict, gt_bbox):
        required = [
            'sgcr_candidate_logits', 'sgcr_candidate_base_boxes',
            'sgcr_candidate_boxes', 'sgcr_final_boxes'
        ]
        if not all(k in pred_dict for k in required):
            zero = gt_bbox.sum() * 0.0
            return zero, zero, zero.detach(), zero.detach(), zero.detach()

        logits = pred_dict['sgcr_candidate_logits']
        base_boxes = pred_dict['sgcr_candidate_base_boxes']
        refined_boxes = pred_dict['sgcr_candidate_boxes']
        final_boxes = pred_dict['sgcr_final_boxes']
        b, k, _ = base_boxes.shape

        gt_cxcywh = box_xywh_to_cxcywh(gt_bbox).clamp(0.0, 1.0)
        gt_expand = gt_cxcywh[:, None, :].expand(b, k, 4)

        with torch.no_grad():
            base_xyxy = box_cxcywh_to_xyxy(base_boxes.detach()).reshape(-1, 4)
            gt_xyxy = box_cxcywh_to_xyxy(gt_expand).reshape(-1, 4)
            iou, _ = box_iou(base_xyxy, gt_xyxy)
            iou = iou.view(b, k).clamp(0.0, 1.0)
            oracle_iou = iou.max(dim=1).values

            center_dist = torch.linalg.vector_norm(
                base_boxes.detach()[..., :2] - gt_expand[..., :2], dim=-1
            )
            # Use IoU quality whenever at least one plausible box exists. If all
            # candidate boxes have zero overlap, fall back to center proximity so
            # semantic candidates can still receive a meaningful learning signal.
            use_iou = oracle_iou >= self.cand_min_oracle_iou
            quality_iou = iou / max(self.cand_quality_temperature, 1e-6)
            quality_center = -center_dist / max(self.cand_quality_temperature, 1e-6)
            quality = torch.where(use_iou[:, None], quality_iou, quality_center)
            target_prob = F.softmax(quality, dim=1)

        cls_loss = -(target_prob * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

        # Geometry supervision follows the same soft candidate ownership.
        l1_per = F.l1_loss(refined_boxes, gt_expand, reduction='none').mean(dim=-1)
        cand_l1 = (target_prob * l1_per).sum(dim=1).mean()

        ref_xyxy = box_cxcywh_to_xyxy(refined_boxes.reshape(-1, 4))
        gt_xyxy = box_cxcywh_to_xyxy(gt_expand.reshape(-1, 4))
        giou, _ = generalized_box_iou(ref_xyxy, gt_xyxy)
        giou_loss_per = (1.0 - giou).view(b, k)
        cand_giou = (target_prob * giou_loss_per).sum(dim=1).mean()
        cand_box_loss = self.loss_weight['l1'] * cand_l1 + self.loss_weight['giou'] * cand_giou

        with torch.no_grad():
            final_iou, _ = box_iou(
                box_cxcywh_to_xyxy(final_boxes.detach()),
                box_cxcywh_to_xyxy(gt_cxcywh),
            )
            selected_iou = final_iou.mean()
            oracle_metric = oracle_iou.mean()

        return cls_loss, cand_box_loss, oracle_metric, selected_iou, target_prob.max(dim=1).values.mean()

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        task_cls_loss = self.objective['task_cls'](
            pred_dict['task_class'], pred_dict['task_class_label']
        )

        gt_bbox = gt_dict['search_anno'][-1]
        gt_gaussian_maps = generate_heatmap(
            gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.ENCODER.STRIDE
        )[-1].unsqueeze(1)

        # Preserve the native SUTrack box objective exactly. SLA-v3 candidate
        # resolution has its own ranking/geometry objectives below.
        native_pred_boxes = pred_dict.get('native_pred_boxes', pred_dict['pred_boxes'])
        if torch.isnan(native_pred_boxes).any() or torch.isnan(pred_dict['pred_boxes']).any():
            raise ValueError('Network outputs contain NaN! Stop Training')

        num_queries = native_pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(native_pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(0.0, 1.0)

        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except Exception:
            giou_loss = torch.tensor(0.0, device=native_pred_boxes.device)
            iou = torch.tensor(0.0, device=native_pred_boxes.device)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        sem_loc_loss, sem_top1, sem_top5 = self._semantic_localization_loss(pred_dict, gt_gaussian_maps)
        id_loc_loss, id_top1, id_top5 = self._identity_localization_loss(pred_dict, gt_gaussian_maps)
        anchor_loss = self._anchor_contrast_loss(pred_dict)
        cand_cls_loss, cand_box_loss, cand_oracle, cand_selected, cand_target_peak = self._candidate_losses(
            pred_dict, gt_bbox
        )

        native_loss = (
            self.loss_weight['giou'] * giou_loss +
            self.loss_weight['l1'] * l1_loss +
            self.loss_weight['focal'] * location_loss +
            self.loss_weight['task_cls'] * task_cls_loss
        )
        loss = (
            native_loss +
            self.sem_loc_weight * sem_loc_loss +
            self.identity_loc_weight * id_loc_loss +
            self.anchor_contrast_weight * anchor_loss +
            self.cand_cls_weight * cand_cls_loss +
            self.cand_box_weight * cand_box_loss
        )

        if return_status:
            mean_iou = iou.detach().mean()
            attn_target = pred_dict.get('sla_target_context_attn', None)
            if attn_target is not None:
                attn_target = attn_target[:, 0].detach().mean()
            else:
                attn_target = torch.tensor(0.0, device=loss.device)
            res_scale = pred_dict.get('sla_identity_res_scale', torch.tensor(0.0, device=loss.device))
            sem_margin = pred_dict.get('sgcr_candidate_semantic_margins', None)
            native_margin = pred_dict.get('sgcr_candidate_native_margins', None)
            sem_margin_mean = sem_margin.detach().mean() if sem_margin is not None else torch.tensor(0.0, device=loss.device)
            native_margin_mean = native_margin.detach().mean() if native_margin is not None else torch.tensor(0.0, device=loss.device)

            status = {
                'Loss/total': loss.item(),
                'Loss/native': native_loss.item(),
                'Loss/giou': giou_loss.item(),
                'Loss/l1': l1_loss.item(),
                'Loss/location': location_loss.item(),
                'Loss/task_class': task_cls_loss.item(),
                'Loss/sem_loc_v1': sem_loc_loss.item(),
                'Loss/identity_loc': id_loc_loss.item(),
                'Loss/anchor_contrast': anchor_loss.item(),
                'Loss/candidate_cls': cand_cls_loss.item(),
                'Loss/candidate_box': cand_box_loss.item(),
                'SLA-v1/top1_hit': sem_top1.item(),
                'SLA-v1/top5_hit': sem_top5.item(),
                'ISIA/top1_hit': id_top1.item(),
                'ISIA/top5_hit': id_top5.item(),
                'ISIA/target_attn': attn_target.item(),
                'ISIA/res_scale': res_scale.detach().mean().item(),
                'SGCR/oracle_iou': cand_oracle.item(),
                'SGCR/selected_iou': cand_selected.item(),
                'SGCR/target_peak': cand_target_peak.item(),
                'SGCR/native_margin': native_margin_mean.item(),
                'SGCR/semantic_margin': sem_margin_mean.item(),
                'IoU': mean_iou.item(),
            }
            return loss, status
        return loss

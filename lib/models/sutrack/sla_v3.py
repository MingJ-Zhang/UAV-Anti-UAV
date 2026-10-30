"""Research-grounded SLA-v3 modules for SUTrack.

Design sources (conceptual, re-implemented for SUTrack):
- ATCTrack (ICCV 2025): explicit target/context modeling.
- CiteTracker (ICCV 2023): image-text correlation for spatial target evidence.
- MVLM (CVPR 2026): distractor-relative VL/classification margins for candidate confidence.

The implementation intentionally starts from the already-positive SLA-v1 path:
raw fused text-to-search semantic localization remains unchanged as an auxiliary
objective. All three structural modules are residual/zero-initialized so an official
SUTrack checkpoint starts from the native prediction path.
"""

import math
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def _gather_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather [B,N,C] tokens with [B,K] indices -> [B,K,C]."""
    b, _, c = x.shape
    idx = indices.unsqueeze(-1).expand(b, indices.size(1), c)
    return x.gather(1, idx)


def _gather_map(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather [B,C,H,W] map values at flattened [B,K] indices -> [B,K,C]."""
    flat = x.flatten(2).transpose(1, 2)
    return _gather_tokens(flat, indices)


def _bbox_token_coverage(bbox_xywh: torch.Tensor, side: int,
                         dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Patch-cell coverage by a normalized xywh box.

    This mirrors the target-mask spirit used by target/context trackers, but computes
    it directly on the template token grid. Output lies in [0,1], shape [B, side^2].
    """
    bbox = bbox_xywh.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    x, y, w, h = bbox.unbind(-1)
    bx1, by1 = x, y
    bx2 = (x + w).clamp(0.0, 1.0)
    by2 = (y + h).clamp(0.0, 1.0)

    coords = torch.arange(side, device=device, dtype=dtype) / float(side)
    x1g, y1g = torch.meshgrid(coords, coords, indexing='xy')
    x1g = x1g.reshape(1, -1)
    y1g = y1g.reshape(1, -1)
    cell = 1.0 / float(side)
    x2g = x1g + cell
    y2g = y1g + cell

    inter_w = (torch.minimum(x2g, bx2.unsqueeze(1)) - torch.maximum(x1g, bx1.unsqueeze(1))).clamp_min(0.0)
    inter_h = (torch.minimum(y2g, by2.unsqueeze(1)) - torch.maximum(y1g, by1.unsqueeze(1))).clamp_min(0.0)
    coverage = (inter_w * inter_h) / (cell * cell)

    # Degenerate / extremely tiny annotation fallback: assign the nearest cell.
    empty = coverage.sum(dim=1) <= 1e-8
    if empty.any():
        cx = (x + 0.5 * w).clamp(0.0, 1.0 - 1e-6)
        cy = (y + 0.5 * h).clamp(0.0, 1.0 - 1e-6)
        ix = torch.floor(cx * side).long().clamp(0, side - 1)
        iy = torch.floor(cy * side).long().clamp(0, side - 1)
        idx = iy * side + ix
        fallback = torch.zeros_like(coverage)
        fallback.scatter_(1, idx.unsqueeze(1), 1.0)
        coverage = torch.where(empty.unsqueeze(1), fallback, coverage)
    return coverage


class InstanceSpecificIdentityAnchor(nn.Module):
    """ISIA: first-frame target/context + language identity anchor.

    A text query attends to two explicitly separated first-template summaries:
    target and context. The resulting visual-language update is injected into the
    native fused text token through a zero-initialized residual scale. Therefore,
    at initialization q_id is exactly normalized native text evidence, preserving
    the SLA-v1 semantic-localization path.
    """

    def __init__(self, dim: int = 512, hidden_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('ISIA hidden_dim must be divisible by num_heads.')
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.norm_text = nn.LayerNorm(dim)
        self.norm_visual = nn.LayerNorm(dim)
        self.text_proj = nn.Linear(dim, hidden_dim)
        self.visual_proj = nn.Linear(dim, hidden_dim)
        self.type_embed = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        nn.init.trunc_normal_(self.type_embed, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.contrast_proj = nn.Linear(dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        # Exact SLA-v1 identity at initialization.
        self.res_scale = nn.Parameter(torch.zeros(1))

    def _pool_target_context(self, static_tokens: torch.Tensor,
                             static_bbox_xywh: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = static_tokens.shape
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise RuntimeError(f'ISIA expects a square template token grid, got N={n}.')

        coverage = _bbox_token_coverage(
            static_bbox_xywh, side, static_tokens.dtype, static_tokens.device
        )
        target_w = coverage / coverage.sum(dim=1, keepdim=True).clamp_min(1e-6)
        context_raw = (1.0 - coverage).clamp_min(0.0)
        context_w = context_raw / context_raw.sum(dim=1, keepdim=True).clamp_min(1e-6)

        target = torch.sum(static_tokens * target_w.unsqueeze(-1), dim=1)
        context = torch.sum(static_tokens * context_w.unsqueeze(-1), dim=1)
        return target, context, coverage

    def forward(self, static_tokens: torch.Tensor, text_token: torch.Tensor,
                static_bbox_xywh: torch.Tensor) -> Dict[str, torch.Tensor]:
        target, context, coverage = self._pool_target_context(static_tokens, static_bbox_xywh)

        q = self.text_proj(self.norm_text(text_token)).unsqueeze(1)
        kv_raw = torch.stack([target, context], dim=1)
        kv = self.visual_proj(self.norm_visual(kv_raw)) + self.type_embed
        cross, attn = self.cross_attn(q, kv, kv, need_weights=True, average_attn_weights=True)

        contrast = self.contrast_proj(self.norm_visual(target - context))
        delta = self.fusion(torch.cat([cross.squeeze(1), contrast], dim=-1))
        scale = torch.tanh(self.res_scale)
        identity = F.normalize(text_token + scale * delta, dim=-1, eps=1e-6)

        return {
            'identity_anchor': identity,
            'target_summary': target,
            'context_summary': context,
            'target_coverage': coverage,
            'target_context_attn': attn.squeeze(1),  # [B,2]
            'identity_res_scale': scale,
        }


class LanguageConditionedSpatialCalibration(nn.Module):
    """LSC: explicit identity-to-search spatial correlation + residual calibration.

    The dense semantic field is a normalized correlation map between q_id and each
    search token. A low-rank interaction adapter injects target-conditioned residual
    features, spatially weighted by that field. The output projection is zero-init,
    so LSC is an exact no-op at initialization.
    """

    def __init__(self, dim: int = 512, hidden_dim: int = 128,
                 temperature: float = 0.15, residual_scale: float = 0.10,
                 gate_max: float = 4.0):
        super().__init__()
        self.temperature = float(temperature)
        self.residual_scale = float(residual_scale)
        self.gate_max = float(gate_max)

        self.norm_x = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.search_proj = nn.Linear(dim, hidden_dim)
        self.query_proj = nn.Linear(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, search_tokens: torch.Tensor,
                identity_anchor: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_norm = self.norm_x(search_tokens)
        q_norm = self.norm_q(identity_anchor)

        semantic_field = F.cosine_similarity(
            F.normalize(x_norm, dim=-1, eps=1e-6),
            F.normalize(q_norm, dim=-1, eps=1e-6).unsqueeze(1),
            dim=-1,
        )
        # Relative spatial weighting; mean approximately 1 before clipping.
        gate = F.softmax(
            semantic_field / max(self.temperature, 1e-6), dim=1
        ) * search_tokens.size(1)
        gate = gate.clamp(max=self.gate_max)

        interaction = F.gelu(
            self.search_proj(x_norm) * self.query_proj(q_norm).unsqueeze(1)
        )
        delta = self.out_proj(interaction)
        calibrated = search_tokens + self.residual_scale * gate.unsqueeze(-1) * delta

        return {
            'calibrated_search': calibrated,
            'semantic_field': semantic_field,
            'spatial_gate': gate,
        }


def _decode_all_cells(score_map: torch.Tensor, size_map: torch.Tensor,
                      offset_map: torch.Tensor) -> torch.Tensor:
    """Decode one candidate box from every CENTER-grid cell -> [B,HW,4] cxcywh."""
    b, _, h, w = score_map.shape
    size = size_map.flatten(2).transpose(1, 2)      # [B,N,2]
    offset = offset_map.flatten(2).transpose(1, 2)  # [B,N,2]
    ys, xs = torch.meshgrid(
        torch.arange(h, device=score_map.device, dtype=size.dtype),
        torch.arange(w, device=score_map.device, dtype=size.dtype),
        indexing='ij',
    )
    xs = xs.reshape(1, -1).expand(b, -1)
    ys = ys.reshape(1, -1).expand(b, -1)
    cx = (xs + offset[..., 0]) / float(w)
    cy = (ys + offset[..., 1]) / float(h)
    return torch.stack([cx, cy, size[..., 0], size[..., 1]], dim=-1)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                        cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _pairwise_iou_batch(boxes: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for [B,N,4] cxcywh boxes -> [B,N,N]."""
    xyxy = _cxcywh_to_xyxy(boxes)
    lt = torch.maximum(xyxy[:, :, None, :2], xyxy[:, None, :, :2])
    rb = torch.minimum(xyxy[:, :, None, 2:], xyxy[:, None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area = ((xyxy[..., 2] - xyxy[..., 0]).clamp_min(0.0) *
            (xyxy[..., 3] - xyxy[..., 1]).clamp_min(0.0))
    union = area[:, :, None] + area[:, None, :] - inter + 1e-8
    return inter / union


def _region_mean_from_grid(tokens: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Mean-pool token-grid features inside candidate boxes.

    Bounds are detached/discretized (as in candidate-region scoring pipelines), while
    gradients propagate to token features. tokens: [B,N,C], boxes: [B,M,4] cxcywh.
    """
    b, n, c = tokens.shape
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise RuntimeError(f'SGCR expects square search token grid, got N={n}.')
    m = boxes.size(1)
    outputs = []
    boxes_d = boxes.detach()

    for bi in range(b):
        feat = tokens[bi].view(side, side, c)
        box = boxes_d[bi]
        cx, cy, bw, bh = box.unbind(-1)
        x1 = torch.floor((cx - 0.5 * bw) * side).long()
        y1 = torch.floor((cy - 0.5 * bh) * side).long()
        x2 = torch.ceil((cx + 0.5 * bw) * side).long()
        y2 = torch.ceil((cy + 0.5 * bh) * side).long()

        x1 = x1.clamp(0, side - 1)
        y1 = y1.clamp(0, side - 1)
        x2 = x2.clamp(1, side)
        y2 = y2.clamp(1, side)
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)

        integral = feat.cumsum(0).cumsum(1)
        integral = F.pad(integral, (0, 0, 1, 0, 1, 0))  # [H+1,W+1,C]
        rect = (integral[y2, x2] - integral[y1, x2] -
                integral[y2, x1] + integral[y1, x1])
        area = ((x2 - x1) * (y2 - y1)).clamp_min(1).to(feat.dtype).unsqueeze(-1)
        outputs.append(rect / area)

    return torch.stack(outputs, dim=0).view(b, m, c)


def _per_box_distractor(boxes: torch.Tensor, scores: torch.Tensor,
                        iou_thr: float) -> torch.Tensor:
    """Strongest non-overlapping distractor score for every candidate.

    Follows the distractor-relative margin principle of MVLM, but is implemented
    batch-wise here for SUTrack's dense CENTER candidates.
    """
    b, n = scores.shape
    iou = _pairwise_iou_batch(boxes)
    outside = iou <= float(iou_thr)
    eye = torch.eye(n, dtype=torch.bool, device=boxes.device).unsqueeze(0)
    outside = outside & (~eye)

    expanded = scores[:, None, :].expand(b, n, n)
    masked = expanded.masked_fill(~outside, -torch.inf)
    distract = masked.max(dim=2).values
    no_valid = ~torch.isfinite(distract)
    if no_valid.any():
        fallback = scores.min(dim=1, keepdim=True).values.expand_as(distract)
        distract = torch.where(no_valid, fallback, distract)
    return distract


class SemanticGeometricCandidateResolution(nn.Module):
    """SGCR: distractor-relative semantic/native margins + learned box resolution.

    All dense CENTER cells form hypotheses. For each hypothesis, semantic and native
    evidence are contrasted against spatially non-overlapping distractors, following
    the margin idea validated by MVLM. A small zero-initialized resolver operates on
    the union of top native and top semantic hypotheses. At initialization, the
    selected box equals the native CENTER maximum (up to fp round-off).
    """

    def __init__(self, dim: int = 512, hidden_dim: int = 128, topk: int = 5,
                 distractor_iou: float = 0.30, box_delta_scale: float = 0.20,
                 dropout: float = 0.0):
        super().__init__()
        self.topk = int(topk)
        self.distractor_iou = float(distractor_iou)
        self.box_delta_scale = float(box_delta_scale)

        # Evidence features: cls, semantic, cls-margin, semantic-margin, area, log-aspect.
        self.score_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)

        self.norm_v = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.visual_proj = nn.Linear(dim, hidden_dim)
        self.query_proj = nn.Linear(dim, hidden_dim)
        self.box_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.box_mlp[-1].weight)
        nn.init.zeros_(self.box_mlp[-1].bias)

    def forward(self, response_map: torch.Tensor, size_map: torch.Tensor,
                offset_map: torch.Tensor, semantic_tokens: torch.Tensor,
                identity_anchor: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = response_map.size(0)
        boxes_all = _decode_all_cells(response_map, size_map, offset_map)
        cls_all = response_map.flatten(1).clamp(1e-4, 1.0 - 1e-4)

        region_feat_all = _region_mean_from_grid(semantic_tokens, boxes_all)
        semantic_all = F.cosine_similarity(
            F.normalize(region_feat_all, dim=-1, eps=1e-6),
            F.normalize(identity_anchor, dim=-1, eps=1e-6).unsqueeze(1),
            dim=-1,
        )

        d_cls = _per_box_distractor(boxes_all, cls_all, self.distractor_iou)
        d_sem = _per_box_distractor(boxes_all, semantic_all, self.distractor_iou)
        margin_cls = torch.tanh(cls_all - d_cls)
        margin_sem = torch.tanh(semantic_all - d_sem)

        k = min(self.topk, cls_all.size(1))
        idx_native = cls_all.topk(k=k, dim=1).indices
        idx_sem = semantic_all.topk(k=k, dim=1).indices
        indices = torch.cat([idx_native, idx_sem], dim=1)  # native first => tie keeps native

        cand_boxes = _gather_tokens(boxes_all, indices)
        cand_region = _gather_tokens(region_feat_all, indices)
        cand_cls = cls_all.gather(1, indices)
        cand_sem = semantic_all.gather(1, indices)
        cand_mcls = margin_cls.gather(1, indices)
        cand_msem = margin_sem.gather(1, indices)

        area = (cand_boxes[..., 2] * cand_boxes[..., 3]).clamp_min(1e-6)
        aspect = torch.log(
            cand_boxes[..., 2].clamp_min(1e-4) /
            cand_boxes[..., 3].clamp_min(1e-4)
        ).clamp(-5.0, 5.0)
        evidence = torch.stack(
            [cand_cls, cand_sem, cand_mcls, cand_msem, area, aspect], dim=-1
        )

        score_delta = self.score_mlp(evidence).squeeze(-1)
        candidate_logits = _inverse_sigmoid(cand_cls) + score_delta

        v = self.visual_proj(self.norm_v(cand_region))
        q = self.query_proj(self.norm_q(identity_anchor)).unsqueeze(1).expand_as(v)
        interaction = v * q
        box_input = torch.cat([v, q, interaction, evidence], dim=-1)
        box_delta = torch.tanh(self.box_mlp(box_input)) * self.box_delta_scale

        # Stable center/scale residual parameterization.
        base_center = cand_boxes[..., :2].clamp(1e-4, 1.0 - 1e-4)
        base_size = cand_boxes[..., 2:].clamp_min(1e-4)
        refined_center = torch.sigmoid(_inverse_sigmoid(base_center) + box_delta[..., :2])
        refined_size = base_size * torch.exp(box_delta[..., 2:].clamp(-2.0, 2.0))
        refined_boxes = torch.cat([refined_center, refined_size], dim=-1)

        best = candidate_logits.argmax(dim=1)
        final_boxes = refined_boxes.gather(
            1, best[:, None, None].expand(b, 1, 4)
        ).squeeze(1)
        final_score = torch.sigmoid(candidate_logits.gather(1, best[:, None]))

        return {
            'final_boxes': final_boxes,
            'final_scores': final_score,
            'candidate_logits': candidate_logits,
            'candidate_indices': indices,
            'candidate_base_boxes': cand_boxes,
            'candidate_boxes': refined_boxes,
            'candidate_native_scores': cand_cls,
            'candidate_semantic_scores': cand_sem,
            'candidate_native_margins': cand_mcls,
            'candidate_semantic_margins': cand_msem,
            'all_native_scores': cls_all,
            'all_semantic_scores': semantic_all,
            'all_native_margins': margin_cls,
            'all_semantic_margins': margin_sem,
        }


class GroundedSLA(nn.Module):
    """Container for ISIA -> LSC -> SGCR."""

    def __init__(self, dim: int, cfg):
        super().__init__()
        hidden = int(getattr(cfg, 'HIDDEN_DIM', 128))
        self.identity_temperature = float(getattr(cfg, 'IDENTITY_TEMPERATURE', 0.10))
        self.isia = InstanceSpecificIdentityAnchor(
            dim=dim,
            hidden_dim=hidden,
            num_heads=int(getattr(cfg, 'ISIA_HEADS', 4)),
            dropout=float(getattr(cfg, 'DROPOUT', 0.0)),
        )
        self.lsc = LanguageConditionedSpatialCalibration(
            dim=dim,
            hidden_dim=hidden,
            temperature=float(getattr(cfg, 'LSC_TEMPERATURE', 0.15)),
            residual_scale=float(getattr(cfg, 'LSC_RESIDUAL_SCALE', 0.10)),
            gate_max=float(getattr(cfg, 'LSC_GATE_MAX', 4.0)),
        )
        self.sgcr = SemanticGeometricCandidateResolution(
            dim=dim,
            hidden_dim=int(getattr(cfg, 'SGCR_HIDDEN_DIM', hidden)),
            topk=int(getattr(cfg, 'SGCR_TOPK', 5)),
            distractor_iou=float(getattr(cfg, 'SGCR_DISTRACTOR_IOU', 0.30)),
            box_delta_scale=float(getattr(cfg, 'SGCR_BOX_DELTA_SCALE', 0.20)),
            dropout=float(getattr(cfg, 'DROPOUT', 0.0)),
        )

    def context(self, search_tokens: torch.Tensor, static_tokens: torch.Tensor,
                text_token: torch.Tensor, static_bbox_xywh: torch.Tensor) -> Dict[str, torch.Tensor]:
        isia = self.isia(static_tokens, text_token, static_bbox_xywh)
        lsc = self.lsc(search_tokens, isia['identity_anchor'])
        out = dict(isia)
        out.update(lsc)
        out['identity_logits'] = lsc['semantic_field'] / max(self.identity_temperature, 1e-6)
        return out

    def resolve(self, response_map: torch.Tensor, size_map: torch.Tensor,
                offset_map: torch.Tensor, semantic_tokens: torch.Tensor,
                identity_anchor: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.sgcr(response_map, size_map, offset_map, semantic_tokens, identity_anchor)

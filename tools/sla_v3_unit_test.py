"""Standalone numerical sanity test for SLA-v3.

Only requires PyTorch. It intentionally imports sla_v3.py by path so the test does
not depend on timm/clip/easydict or UAV-Anti-UAV data.
"""
from pathlib import Path
import importlib.util
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / 'lib' / 'models' / 'sutrack' / 'sla_v3.py'
spec = importlib.util.spec_from_file_location('sla_v3_standalone', MOD)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    torch.manual_seed(7)
    B, C = 2, 512
    NS, NZ = 14 * 14, 7 * 7

    search = torch.randn(B, NS, C, requires_grad=True)
    static = torch.randn(B, NZ, C, requires_grad=True)
    text = torch.randn(B, C, requires_grad=True)
    bbox = torch.tensor([[0.32, 0.28, 0.22, 0.18],
                         [0.48, 0.41, 0.10, 0.12]], dtype=torch.float32)

    isia = mod.InstanceSpecificIdentityAnchor(C, hidden_dim=128, num_heads=4)
    out_i = isia(static, text, bbox)
    expected_q = F.normalize(text, dim=-1, eps=1e-6)
    id_diff = (out_i['identity_anchor'] - expected_q).abs().max().item()
    assert id_diff < 1e-6, id_diff

    # ISIA must be able to leave the zero-init state through its residual scale.
    id_logits = F.cosine_similarity(
        F.normalize(search, dim=-1),
        out_i['identity_anchor'].unsqueeze(1), dim=-1
    ) / 0.1
    loss_i = F.cross_entropy(id_logits, torch.tensor([17, 96]))
    loss_i.backward(retain_graph=True)
    isia_scale_grad = float(isia.res_scale.grad.abs().item())
    assert isia_scale_grad > 0, isia_scale_grad

    lsc = mod.LanguageConditionedSpatialCalibration(C, hidden_dim=128)
    out_l = lsc(search, out_i['identity_anchor'].detach())
    lsc_diff = (out_l['calibrated_search'] - search).abs().max().item()
    assert lsc_diff == 0.0, lsc_diff
    loss_l = out_l['calibrated_search'].square().mean()
    loss_l.backward(retain_graph=True)
    lsc_grad = float(lsc.out_proj.weight.grad.abs().sum().item())
    assert lsc_grad > 0, lsc_grad

    response = torch.sigmoid(torch.randn(B, 1, 14, 14))
    size_map = torch.sigmoid(torch.randn(B, 2, 14, 14) - 2.0)
    offset_map = torch.zeros(B, 2, 14, 14)

    sgcr = mod.SemanticGeometricCandidateResolution(C, hidden_dim=128, topk=5)
    out_s = sgcr(response, size_map, offset_map, search.detach(), expected_q.detach())
    all_boxes = mod._decode_all_cells(response, size_map, offset_map)
    native_idx = response.flatten(1).argmax(dim=1)
    native_box = all_boxes.gather(1, native_idx[:, None, None].expand(B, 1, 4)).squeeze(1)
    sgcr_diff = (out_s['final_boxes'] - native_box).abs().max().item()
    assert sgcr_diff < 1e-6, sgcr_diff

    # Score and box output heads must receive gradient immediately.
    target_candidate = torch.tensor([1, 3])
    loss_score = F.cross_entropy(out_s['candidate_logits'], target_candidate)
    loss_box = out_s['candidate_boxes'].square().mean()
    (loss_score + loss_box).backward()
    score_grad = float(sgcr.score_mlp[-1].weight.grad.abs().sum().item())
    box_grad = float(sgcr.box_mlp[-1].weight.grad.abs().sum().item())
    assert score_grad > 0, score_grad
    assert box_grad > 0, box_grad

    params = sum(p.numel() for p in isia.parameters()) + \
             sum(p.numel() for p in lsc.parameters()) + \
             sum(p.numel() for p in sgcr.parameters())

    print('ISIA init identity max diff:', id_diff)
    print('LSC init feature max diff:', lsc_diff)
    print('SGCR init bbox max diff:', sgcr_diff)
    print('ISIA residual-scale grad:', isia_scale_grad)
    print('LSC zero-output grad:', lsc_grad)
    print('SGCR score-head grad:', score_grad)
    print('SGCR box-head grad:', box_grad)
    print('SLA-v3 new parameters:', params)
    print('SLA-v3 unit test: PASS')


if __name__ == '__main__':
    main()

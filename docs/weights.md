# Weights

## Training initialization

| File | Component | Source |
|:--|:--|:--|
| `fast_itpn_base_clipl_e1600.pt` | Fast-iTPN-B visual initialization | [SUTrack training instructions](https://github.com/chenxin-dlut/SUTrack#training) |
| `SUTRACK_ep0180.pth.tar` | Full SUTrack-B224 tracking initialization | [SUTrack model collection](https://huggingface.co/xche32/SUTrack) |
| CLIP ViT-L/14 | Text encoder | Downloaded by the [CLIP loader](https://github.com/openai/CLIP) at model construction |

Place the first two files in `pretrained/`. Their paths are set in `experiments/sutrack/sutrack_b224_sla_v3_uavanti.yaml`, under `MODEL.ENCODER.PRETRAIN_TYPE` and `TRAIN.PRETRAINED_PATH`.

ISIA, LSC, and SGCR use residual or output-layer zero initialization. The first training step therefore starts with the native semantic identity, search features, and CENTER candidate prediction, while the added layers receive gradients for adaptation.

## Trained SLA-Track checkpoint

The epoch-40 checkpoint produced by the training command is:

```text
outputs/checkpoints/train/sutrack/sutrack_b224_sla_v3_uavanti/SUTRACK_ep0040.pth.tar
```

Use `--checkpoint /path/to/SUTRACK_ep0040.pth.tar` with `tracking/test.py`. The online tracker loads the complete network state, including all three SLA-Track modules.

Checkpoint tensors retain the production implementation's parameter names for compatibility with existing experiments.

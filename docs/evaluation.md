# Tracking & Evaluation

Run from the repository root with the dataset paths configured by `tools/setup_paths.py`.

## UAV-Anti-UAV

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/test.py sutrack sutrack_b224_sla_v3_uavanti \
  --dataset_name uavantiuav \
  --checkpoint outputs/checkpoints/train/sutrack/sutrack_b224_sla_v3_uavanti/SUTRACK_ep0040.pth.tar \
  --threads 0 --num_gpus 1
```

Add `--sequence 000113` for an individual sequence. The complete test split has 420 sequences. Prediction files are written beneath `outputs/test/tracking_results/sutrack/sutrack_b224_sla_v3_uavanti/`, one `xywh` row per frame.

## Zero-shot transfer

Use the same UAV-Anti-UAV-trained checkpoint and the fixed language prompt on both datasets:

```bash
GEN_NLP_TEXT="drone flying in the sky" CUDA_VISIBLE_DEVICES=0 \
python tracking/test.py sutrack sutrack_b224_sla_v3_uavanti \
  --dataset_name a2atracking --checkpoint /path/to/SUTRACK_ep0040.pth.tar \
  --threads 0 --num_gpus 1

GEN_NLP_TEXT="drone flying in the sky" CUDA_VISIBLE_DEVICES=0 \
python tracking/test.py sutrack sutrack_b224_sla_v3_uavanti \
  --dataset_name dutantiuav --checkpoint /path/to/SUTRACK_ep0040.pth.tar \
  --threads 0 --num_gpus 1
```

Set `GEN_RESULTS_PATH=outputs/a2a` or `GEN_RESULTS_PATH=outputs/dut` on these commands to store each benchmark in a separate directory. A2A-UAV-Tracking has 23 test sequences; DUT Anti-UAV has 20.

## Score saved predictions

```bash
python tools/evaluate.py \
  --dataset uavantiuav \
  --results outputs/test/tracking_results/sutrack/sutrack_b224_sla_v3_uavanti \
  --gt-root /path/to/UAV-Anti-UAV/Test \
  --output outputs/reports/uavantiuav.json
```

For A2A, use `--dataset a2atracking --gt-root /path/to/A2ATracking`; for DUT, use `--dataset dutantiuav --gt-root /path/to/Anti-UAV-Tracking-V0GT`. Point `--results` to the corresponding benchmark's prediction directory.

The evaluator computes each curve per sequence and then averages sequences with equal weight:

| Metric | Definition |
|:--|:--|
| Success | Mean success rate over 21 IoU thresholds from 0 to 1; strict `IoU > threshold`. |
| Prec@20 | Fraction of frames with center error ≤20 pixels. |
| NP AUC | Mean normalized-precision curve over thresholds 0, 0.01, …, 0.50. |
| SR@0.5 | Fraction of frames with `IoU > 0.5`. |

Initialization predictions are replaced by the first-frame ground truth. Normalized center errors use the ground-truth width and height. The JSON report contains all aggregate curves and per-sequence metrics. Missing predictions and frame-count mismatches are reported as errors.

## Inference settings

Search factor 4.0; search size 224; template size 112; two templates. CENTER responses are Hann-windowed before SGCR. Dynamic-template updates use native confidence at an interval of 25 frames and threshold 0.70.

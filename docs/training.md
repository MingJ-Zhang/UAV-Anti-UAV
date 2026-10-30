# Training

Run commands from the repository root after [installing dependencies](../README.md#1-install), [configuring data](data.md), and [preparing initialization weights](weights.md).

## Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/train.py \
  --config sutrack_b224_sla_v3_uavanti \
  --save_dir outputs --mode single --seed 42
```

The launcher uses the active Python interpreter and passes the selected seed to the training runner. Add `--dry-run` to inspect its command.

## Multiple GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py \
  --config sutrack_b224_sla_v3_uavanti \
  --save_dir outputs --mode multiple --nproc_per_node 2 --seed 42
```

Distributed training uses `torch.distributed.run`. `TRAIN.BATCH_SIZE` is the batch size per process. Set it to `16 / number_of_GPUs` to retain the paper's total batch size of 16.

## Main configuration

Configuration: [`sutrack_b224_sla_v3_uavanti.yaml`](../experiments/sutrack/sutrack_b224_sla_v3_uavanti.yaml).

| Setting | Value |
|:--|:--|
| Optimizer | AdamW |
| Epochs / samples per epoch | 40 / 20,000 |
| Batch size (single GPU) | 16 |
| Learning rate / backbone multiplier | 5e-5 / 0.1 |
| LR decay | ×0.1 at epoch 30 |
| Weight decay | 1e-4 |
| Search / template resolution | 224 / 112 |
| Templates | First-frame reference + dynamic template |
| Text encoder | Frozen CLIP ViT-L/14 |
| Shared interaction dimension | 128 |
| Candidate pool | Native Top-5 + semantic Top-5 |

## Objectives and logs

The actor in [`lib/train/actors/sutrack.py`](../lib/train/actors/sutrack.py) combines native CENTER localization and task classification with dense semantic alignment, identity localization, anchor contrast, candidate classification, and candidate box regression.

Training logs include `Loss/sem_loc_v1`, `Loss/identity_loc`, `Loss/anchor_contrast`, `Loss/candidate_cls`, and `Loss/candidate_box`, plus module diagnostics. The `sem_loc_v1` key identifies the dense semantic-localization term in the checkpoint-compatible implementation.

Checkpoints, logs, and TensorBoard events are written beneath `--save_dir`. Reusing a run directory resumes from its latest checkpoint; choose a new directory to start another run.

```bash
tensorboard --logdir outputs/tensorboard
```

Use the same `--save-dir` value in `tools/setup_paths.py` so the default evaluator resolves the matching checkpoint and output directories.

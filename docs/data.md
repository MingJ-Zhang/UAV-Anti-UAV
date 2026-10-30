# Data Preparation

SLA-Track trains on UAV-Anti-UAV and evaluates on three RGB tracking benchmarks.

## Directory layouts

```text
datasets/
├── UAV-Anti-UAV/
│   ├── Train/<sequence>/
│   │   ├── imgs/000001.jpg
│   │   ├── groundtruth_rect.txt
│   │   └── language.txt
│   └── Test/<sequence>/
│       ├── imgs/000001.jpg
│       ├── groundtruth_rect.txt
│       └── language.txt
├── A2ATracking/<sequence>/
│   ├── img/00000001.jpg
│   └── groundtruth.txt
└── DUT-Anti-UAV/
    ├── Anti-UAV-Tracking-V0/<sequence>/000001.jpg
    └── Anti-UAV-Tracking-V0GT/<sequence>_gt.txt
```

Each annotation row is `x, y, width, height` in image pixels. The tracking adapters accept comma- or whitespace-separated annotations. The UAV-Anti-UAV adapter also recognizes `images`, `img`, and `color` image folders.

## Configure locations

```bash
python tools/setup_paths.py --data-dir /path/to/datasets --save-dir outputs
```

For datasets stored separately:

```bash
python tools/setup_paths.py \
  --data-dir /path/to/datasets \
  --uavanti-root /path/to/UAV-Anti-UAV-imgs \
  --a2a-root /path/to/A2ATracking \
  --dut-images /path/to/Anti-UAV-Tracking-V0 \
  --dut-gt /path/to/Anti-UAV-Tracking-V0GT \
  --save-dir /path/to/outputs
```

The command writes machine-specific settings to `lib/train/admin/local.py` and `lib/test/evaluation/local.py`. These files are ignored by Git. Dataset contents are read in place; checkpoints and predictions go to the selected output directory.

## Language protocol

- **UAV-Anti-UAV:** the first line of each sequence's `language.txt` supplies the description for training and evaluation.
- **A2A-UAV-Tracking and DUT Anti-UAV:** use `GEN_NLP_TEXT="drone flying in the sky"` during inference to reproduce the paper's shared transfer prompt.

The official test sets contain 420, 23, and 20 sequences, respectively. See [Evaluation](evaluation.md) for the corresponding commands.

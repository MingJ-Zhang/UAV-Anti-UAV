#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
exec "${PYTHON:-python}" tracking/train.py --save_dir "${SLA_TRACK_SAVE_DIR:-outputs}" "$@"

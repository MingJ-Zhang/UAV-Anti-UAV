"""Launch SLA-Track training with the active Python environment."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', default='sutrack')
    parser.add_argument('--config', default='sutrack_b224_sla_v3_uavanti')
    parser.add_argument('--save_dir', default='outputs')
    parser.add_argument('--mode', choices=['single', 'multiple'], default='single')
    parser.add_argument('--nproc_per_node', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)
    parser.add_argument('--dry-run', action='store_true', help='Print the command without starting training.')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.nproc_per_node < 1:
        raise SystemExit('--nproc_per_node must be positive.')
    command = [sys.executable]
    if args.mode == 'multiple':
        command += ['-m', 'torch.distributed.run', '--standalone',
                    '--nproc_per_node', str(args.nproc_per_node)]
    command += [str(root / 'lib/train/run_training.py'), '--script', args.script,
                '--config', args.config, '--save_dir', str(Path(args.save_dir).resolve()),
                '--seed', str(args.seed), '--use_lmdb', str(args.use_lmdb)]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()

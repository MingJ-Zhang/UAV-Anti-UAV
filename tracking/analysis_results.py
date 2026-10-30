"""Evaluate saved tracking predictions with the benchmark OPE protocol."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/evaluate.py'), run_name='__main__')

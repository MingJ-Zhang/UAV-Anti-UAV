"""Check portable setup, launcher arguments, and the OPE scoring protocol."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_module('setup_paths', ROOT / 'tools/setup_paths.py')
scoring = load_module('scoring', ROOT / 'tools/evaluate.py')


class ReleaseTests(unittest.TestCase):
    def test_paths_with_spaces_and_quotes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project's code"
            data = Path(directory) / 'data root'
            output = Path(directory) / 'training output'
            train, test = setup.write_settings(project, data, output)
            settings = load_module('train_paths', train).EnvironmentSettings()
            self.assertEqual(settings.uavanti_dir, str(data / 'UAV-Anti-UAV'))
            self.assertEqual(settings.workspace_dir, str(output))
            compile(test.read_text(), str(test), 'exec')
            self.assertIn(repr(str(output / 'test/tracking_results')), test.read_text())

    def test_launcher_uses_active_interpreter_and_preserves_arguments(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tracking/train.py'), '--dry-run',
                                 '--save_dir', 'output with spaces', '--seed', '44',
                                 '--mode', 'multiple', '--nproc_per_node', '2'],
                                text=True, capture_output=True, check=True)
        import shlex
        command = shlex.split(result.stdout.strip())
        self.assertEqual(command[0], sys.executable)
        self.assertIn('torch.distributed.run', command)
        self.assertEqual(command[command.index('--save_dir') + 1], str(Path('output with spaces').resolve()))
        self.assertEqual(command[command.index('--seed') + 1], '44')

    def test_initialization_and_strict_iou_thresholds(self):
        truth = np.array([[0., 0., 10., 10.], [0., 0., 10., 10.]])
        prediction = np.array([[100., 100., 10., 10.], [0., 0., 5., 10.]])
        success, precision, normalized = scoring.sequence_curves(prediction, truth)
        self.assertEqual(success[10], 0.5)  # IoU == 0.5 is excluded.
        self.assertEqual(success[-1], 0.)
        self.assertEqual(precision[0], 0.5)  # Initialization is replaced by GT.
        self.assertEqual(normalized[25], 1.)  # Center thresholds are inclusive.

    def test_equal_sequence_weighting_and_missing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / 'results'
            results.mkdir()
            for name, length in [('short', 2), ('long', 10)]:
                seq = root / 'gt' / name
                seq.mkdir(parents=True)
                truth = np.tile([0., 0., 10., 10.], (length, 1))
                prediction = truth.copy()
                if name == 'long':
                    prediction[1:, :2] = 100
                np.savetxt(seq / 'groundtruth_rect.txt', truth)
                np.savetxt(results / (name + '.txt'), prediction)
            report = scoring.evaluate(results, root / 'gt', 'uavantiuav')
            self.assertEqual(report['n_eval'], 2)
            self.assertAlmostEqual(report['precision_20px'], 0.55)
            (results / 'short.txt').unlink()
            with self.assertRaises(FileNotFoundError):
                scoring.evaluate(results, root / 'gt', 'uavantiuav')

    def test_frame_mismatch_is_reported(self):
        with self.assertRaises(ValueError):
            scoring.sequence_curves(np.ones((2, 4)), np.ones((3, 4)))


if __name__ == '__main__':
    unittest.main()

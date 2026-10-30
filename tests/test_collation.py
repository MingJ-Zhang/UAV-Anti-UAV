"""Exercise the inherited tracking batch format on current Python and PyTorch."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('tracking_loader', ROOT / 'lib/train/data/loader.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class CollationTests(unittest.TestCase):
    def test_nested_batch_preserves_language_and_frame_axis(self):
        samples = [dict(images=torch.ones(2, 3, 4, 4) * i, language='target {}'.format(i),
                        meta=dict(index=i)) for i in range(2)]
        batch = loader.ltr_collate_stack1(samples)
        self.assertEqual(tuple(batch['images'].shape), (2, 2, 3, 4, 4))
        self.assertEqual(batch['language'], ['target 0', 'target 1'])
        self.assertEqual(batch['meta']['index'].tolist(), [0, 1])

    def test_numpy_arrays_and_scalars(self):
        for collate in [loader.ltr_collate, loader.ltr_collate_stack1]:
            self.assertEqual(collate([np.float32(1), np.float32(2)]).tolist(), [1, 2])
            self.assertEqual(tuple(collate([np.ones((2, 3)), np.zeros((2, 3))]).shape), (2, 2, 3))
            with self.assertRaises(TypeError):
                collate([np.array(['language']), np.array(['text'])])


if __name__ == '__main__':
    unittest.main()

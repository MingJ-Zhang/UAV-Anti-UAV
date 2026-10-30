import os
import re
import numpy as np
import pandas as pd
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList


class UAVAntiUAVDataset(BaseDataset):
    IMG_DIRS = ('imgs', 'images', 'img', 'color')
    GT_FILES = ('groundtruth_rect.txt', 'groundtruth.txt', 'groundtruth_rect.1.txt')
    NLP_FILES = ('language.txt', 'nlp.txt', 'description.txt')

    def __init__(self):
        super().__init__()
        root = getattr(self.env_settings, 'uavanti_path', '')
        self.base_path = self._resolve(root)
        self.sequence_list = sorted([d for d in os.listdir(self.base_path)
                                     if os.path.isdir(os.path.join(self.base_path, d))])

    @staticmethod
    def _resolve(root):
        for p in (os.path.join(root, 'Test'), os.path.join(root, 'test'), root):
            if p and os.path.isdir(p):
                dirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
                if dirs:
                    return p
        raise FileNotFoundError('Set uavanti_path in lib/test/evaluation/local.py')

    @staticmethod
    def _natural_key(name):
        return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', name)]

    def _find(self, seq, names):
        for n in names:
            p = os.path.join(seq, n)
            if os.path.exists(p): return p
        raise FileNotFoundError('%s under %s' % (names, seq))

    def _construct_sequence(self, name):
        seq = os.path.join(self.base_path, name)
        gt_path = self._find(seq, self.GT_FILES)
        gt = pd.read_csv(gt_path, sep=r'[,\s]+', engine='python', header=None,
                         dtype=np.float64).values[:, :4]
        img_dir = self._find(seq, self.IMG_DIRS)
        ims = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        frames = [os.path.join(img_dir, f) for f in sorted(ims, key=self._natural_key)]
        language = ''
        for n in self.NLP_FILES:
            p = os.path.join(seq, n)
            if os.path.isfile(p):
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    language = f.readline().strip()
                break
        target_visible = ((gt[:, 2] > 0) & (gt[:, 3] > 0)).astype(np.uint8)
        return Sequence(name, frames, 'uavantiuaV', gt.reshape(-1, 4),
                        target_visible=target_visible, language_query=language)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)

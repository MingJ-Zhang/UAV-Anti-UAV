import os
import re
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from lib.train.admin import env_settings
from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset


class UAVAntiVL(BaseVideoDataset):
    """UAV-Anti-UAV RGB-language training adapter.

    Supported layouts include either:
      root/Train/<sequence>/{imgs,groundtruth_rect.txt,language.txt}
    or a root that already points to the training split.
    """
    IMG_DIRS = ('imgs', 'images', 'img', 'color')
    GT_FILES = ('groundtruth_rect.txt', 'groundtruth.txt', 'groundtruth_rect.1.txt')
    NLP_FILES = ('language.txt', 'nlp.txt', 'description.txt')

    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train', data_fraction=None,
                 multi_modal_vision=False, multi_modal_language=True, use_nlp=True):
        root = getattr(env_settings(), 'uavanti_dir', '') if root is None else root
        split_root = self._resolve_split_root(root, split)
        super().__init__('UAVANTIUAV', split_root, image_loader)
        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language
        self.use_nlp = use_nlp
        self.sequence_list = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])
        if data_fraction is not None:
            random.seed(0)
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))
        if not self.sequence_list:
            raise RuntimeError('No UAV-Anti-UAV sequences found under %s' % self.root)

    @staticmethod
    def _resolve_split_root(root, split):
        candidates = [root]
        if split.lower() == 'train':
            candidates = [os.path.join(root, 'Train'), os.path.join(root, 'train'), root]
        elif split.lower() == 'test':
            candidates = [os.path.join(root, 'Test'), os.path.join(root, 'test'), root]
        for p in candidates:
            if p and os.path.isdir(p):
                # choose the first folder that actually contains sequence directories
                dirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
                if dirs:
                    return p
        raise FileNotFoundError('Cannot resolve UAV-Anti-UAV %s split from %s' % (split, root))

    def get_name(self):
        return 'UAVANTIUAV'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _seq_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    @staticmethod
    def _find_existing(base, names):
        for n in names:
            p = os.path.join(base, n)
            if os.path.exists(p):
                return p
        raise FileNotFoundError('None of %s found under %s' % (names, base))

    def _read_bbox(self, seq_path):
        path = self._find_existing(seq_path, self.GT_FILES)
        # Accept comma, tab, or whitespace separated xywh.
        arr = pd.read_csv(path, sep=r'[,\s]+', engine='python', header=None,
                          dtype=np.float32, comment='#').values
        arr = arr[:, :4]
        return torch.tensor(arr, dtype=torch.float32)

    def _read_nlp(self, seq_path):
        for name in self.NLP_FILES:
            p = os.path.join(seq_path, name)
            if os.path.isfile(p):
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.readline().strip()
                return text
        return ''

    def _image_dir(self, seq_path):
        return self._find_existing(seq_path, self.IMG_DIRS)

    @staticmethod
    def _natural_key(name):
        return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', name)]

    def _images(self, seq_path):
        d = self._image_dir(seq_path)
        files = [f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
        return [os.path.join(d, f) for f in sorted(files, key=self._natural_key)]

    def get_sequence_info(self, seq_id):
        seq_path = self._seq_path(seq_id)
        bbox = self._read_bbox(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone()
        out = {'bbox': bbox, 'valid': valid, 'visible': visible}
        if self.multi_modal_language and self.use_nlp:
            out['nlp'] = self._read_nlp(seq_path)
        return out

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._seq_path(seq_id)
        images = self._images(seq_path)
        frames = []
        for fid in frame_ids:
            frame = self.image_loader(images[fid])
            if self.multi_modal_vision and frame.shape[-1] == 3:
                frame = np.concatenate((frame, frame), axis=-1)
            frames.append(frame)
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {}
        for key, value in anno.items():
            if key == 'nlp':
                anno_frames[key] = [value for _ in frame_ids]
            else:
                anno_frames[key] = [value[fid, ...].clone() for fid in frame_ids]
        meta = OrderedDict({'object_class_name': 'uav', 'motion_class': None,
                            'major_class': 'uav', 'root_class': 'uav', 'motion_adverb': None})
        return frames, anno_frames, meta

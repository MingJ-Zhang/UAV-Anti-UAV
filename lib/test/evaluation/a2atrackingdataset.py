"""A2ATracking test adapter. Original data is read-only."""
from __future__ import print_function

import inspect
import os

import numpy as np
from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList

A2A_ROOT = 'data/A2ATracking'
DEFAULT_NLP = "drone flying in the sky"
DATASET_NAME = "a2atracking"


def _nlp_text(seq_dir):
    forced = os.environ.get("GEN_NLP_TEXT")
    if forced is not None and str(forced).strip():
        return str(forced).strip()
    nlp_path = os.path.join(seq_dir, "language.txt")
    if os.path.isfile(nlp_path):
        with open(nlp_path, "r") as f:
            text = f.read().strip()
            if text:
                return text
    return DEFAULT_NLP


def _load_xywh(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().replace(",", " ")
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) >= 4:
                rows.append(vals[:4])
    if not rows:
        raise RuntimeError("empty gt: %s" % path)
    return np.asarray(rows, dtype=np.float64)


def _list_frames(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    names = [n for n in os.listdir(folder) if os.path.splitext(n)[1].lower() in exts]
    names.sort()
    return [os.path.join(folder, n) for n in names]


def _make_sequence(name, frames, gt, text):
    kwargs = {}
    sig = inspect.signature(Sequence.__init__)
    if "language_query" in sig.parameters:
        kwargs["language_query"] = text
    if "language" in sig.parameters:
        kwargs["language"] = text
    if "text_description" in sig.parameters:
        kwargs["text_description"] = text
    seq = Sequence(
        name,
        frames,
        DATASET_NAME,
        gt.reshape(-1, 4),
        object_class="uav",
        **kwargs
    )
    seq.language = text
    if not hasattr(seq, "language_query"):
        seq.language_query = text
    return seq


class A2ATrackingDataset(BaseDataset):
    def __init__(self, split="test"):
        super(A2ATrackingDataset, self).__init__()
        self.split = split
        self.base_path = os.environ.get("A2A_ROOT", getattr(self.env_settings, "a2atracking_path", A2A_ROOT))
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        seqs = []
        for name in sorted(os.listdir(self.base_path)):
            seq_dir = os.path.join(self.base_path, name)
            img_dir = os.path.join(seq_dir, "img")
            gt = os.path.join(seq_dir, "groundtruth.txt")
            if os.path.isdir(img_dir) and os.path.isfile(gt):
                seqs.append(name)
        if not seqs:
            raise RuntimeError("no A2ATracking sequences under %s" % self.base_path)
        return seqs

    def _construct_sequence(self, sequence_name):
        seq_dir = os.path.join(self.base_path, sequence_name)
        frames = _list_frames(os.path.join(seq_dir, "img"))
        gt = _load_xywh(os.path.join(seq_dir, "groundtruth.txt"))
        n = min(len(frames), len(gt))
        frames = frames[:n]
        gt = gt[:n]
        text = _nlp_text(seq_dir)
        return _make_sequence(sequence_name, frames, gt, text)

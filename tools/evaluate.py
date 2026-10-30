"""Compute sequence-averaged one-pass tracking metrics from saved xywh boxes."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_boxes(path):
    try:
        boxes = np.loadtxt(path, delimiter=',', ndmin=2)
    except ValueError:
        boxes = np.loadtxt(path, ndmin=2)
    if boxes.shape[0] == 0 or boxes.shape[1] < 4 or not np.isfinite(boxes[:, :4]).all():
        raise ValueError('Expected finite xywh rows: {}'.format(path))
    return boxes[:, :4].astype(np.float64)


def sequence_curves(prediction, truth):
    if prediction.shape != truth.shape:
        raise ValueError('Prediction and ground-truth frame counts must match.')
    prediction = prediction.copy()
    prediction[0] = truth[0]
    intersection = np.maximum(
        np.minimum(prediction[:, :2] + prediction[:, 2:], truth[:, :2] + truth[:, 2:])
        - np.maximum(prediction[:, :2], truth[:, :2]), 0).prod(axis=1)
    union = prediction[:, 2:].prod(axis=1) + truth[:, 2:].prod(axis=1) - intersection
    iou = np.clip(intersection / (union + np.finfo(float).eps), 0, 1)
    pred_center = prediction[:, :2] + (prediction[:, 2:] - 1) / 2
    gt_center = truth[:, :2] + (truth[:, 2:] - 1) / 2
    delta = pred_center - gt_center
    error = np.sqrt((delta ** 2).sum(axis=1))
    denominator = truth[:, 2:] + 1e-16
    normalized = np.sqrt(((pred_center / denominator - gt_center / denominator) ** 2).sum(axis=1))
    return (np.mean(iou[:, None] > np.linspace(0, 1, 21), axis=0),
            np.mean(error[:, None] <= np.arange(51), axis=0),
            np.mean(normalized[:, None] <= np.arange(51) / 100, axis=0))


def ground_truth_files(root, dataset):
    root = Path(root)
    if dataset == 'uavantiuav':
        return [(p.parent.name, p) for p in sorted(root.glob('*/groundtruth_rect.txt'))]
    if dataset == 'a2atracking':
        return [(p.parent.name, p) for p in sorted(root.glob('*/groundtruth.txt'))]
    return [(p.stem[:-3], p) for p in sorted(root.glob('*_gt.txt'))]


def evaluate(results, gt_root, dataset):
    truths = ground_truth_files(gt_root, dataset)
    if not truths:
        raise ValueError('No ground-truth sequences found in {}'.format(gt_root))
    predictions = {}
    for path in sorted(Path(results).rglob('*.txt')):
        if path.name.endswith(('_time.txt', '_all_boxes.txt', '_all_scores.txt')):
            continue
        if path.stem in predictions:
            raise ValueError('Multiple predictions for sequence {}'.format(path.stem))
        predictions[path.stem] = path
    per_sequence, all_curves = {}, []
    for name, path in truths:
        if name not in predictions:
            raise FileNotFoundError('Missing prediction: {}.txt'.format(name))
        curves = sequence_curves(load_boxes(predictions[name]), load_boxes(path))
        all_curves.append(curves)
        per_sequence[name] = dict(success_auc=float(curves[0].mean()),
                                  precision_20px=float(curves[1][20]),
                                  norm_precision_auc=float(curves[2].mean()),
                                  **{'success_rate_0.5': float(curves[0][10])})
    success, precision, normalized = [np.mean([v[i] for v in all_curves], axis=0) for i in range(3)]
    return dict(dataset=dataset, n_eval=len(truths),
                success_auc=float(success.mean()), precision_20px=float(precision[20]),
                norm_precision_auc=float(normalized.mean()),
                **{'success_rate_0.5': float(success[10])},
                curves=dict(success=success.tolist(), precision=precision.tolist(),
                            normalized_precision=normalized.tolist()), per_sequence=per_sequence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['uavantiuav', 'a2atracking', 'dutantiuav'], required=True)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--gt-root', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = evaluate(args.results, args.gt_root, args.dataset)
    print('Evaluated {} sequences'.format(report['n_eval']))
    for key in ['success_auc', 'precision_20px', 'norm_precision_auc', 'success_rate_0.5']:
        print('{}: {:.4f}'.format(key, report[key]))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('Report:', args.output)


if __name__ == '__main__':
    main()

"""Reproduce the frozen ReImageNet label-auditing population from author data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_SHA256 = 'e25897e79ce5bf0e47a1593e702b75cba70b8409861f93e94af6f47f7d7f91ea'
LABELS_SHA256 = 'b31cfdf49f8bbbd361fe7742d3acb243a06a68659e6ed1721b460ae8857a4ced'
CLASS_FILE = Path(__file__).with_name('f1_class_ids.json')


def prepare(source: Path) -> dict[str, np.ndarray]:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Source checksum mismatch. Obtain the pinned revision in README.md; do not normalize or reorder the file.')
    submitted, verified, record_ids = [], [], []
    for line in raw.decode('utf-8').splitlines():
        record = json.loads(line)
        y, v = record['original_class'], record['reannotated_labels']
        if len(y) == len(v) == 1 and 0 <= y[0] < 1000 and 0 <= v[0] < 1000:
            submitted.append(y[0])
            verified.append(v[0])
            record_ids.append(record['image_name'])
    y = np.asarray(submitted, dtype=np.int64)
    v = np.asarray(verified, dtype=np.int64)
    ids = np.asarray(record_ids)
    target = np.asarray(json.loads(CLASS_FILE.read_text()), dtype=np.int64)
    if len(target) != 351 or not np.array_equal(target, np.unique(target)) or target.min() < 0 or target.max() > 999:
        raise ValueError('Invalid fixed F1 class list.')
    label_hash = hashlib.sha256(np.column_stack((y, v)).astype('<i8').tobytes()).hexdigest()
    if label_hash != LABELS_SHA256:
        raise ValueError('Ordered label pairs differ from the experiment population.')
    a, b = np.isin(y, target), np.isin(v, target)
    f1 = float(2 * ((y == v) & a).sum() / (a.sum() + b.sum()))
    if (len(y), len(np.unique(ids)), int((y != v).sum()), len(np.unique(y)), len(np.unique(v))) != (30908, 30908, 2501, 964, 958):
        raise ValueError('Population count validation failed.')
    if (int(a.sum()), int(b.sum())) != (9584, 9682) or not np.isclose(f1, 0.9293055123014637, rtol=0, atol=1e-15):
        raise ValueError('F1 target validation failed.')
    return dict(submitted=y, verified=v, record_id=ids,
                tags=np.asarray([str(i) for i in range(1000)]), f1_class_ids=target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Pinned author reannotation.jsonl')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('paired_labels.npz'))
    args = parser.parse_args()
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    if args.output.exists():
        parser.error('Output already exists; choose a new path.')
    arrays = prepare(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as destination:
        np.savez_compressed(destination, **arrays)
    with np.load(args.output, allow_pickle=False) as saved:
        for key, values in arrays.items():
            np.testing.assert_array_equal(saved[key], values)
    print(f'Validated and wrote 30,908 records, 2,501 mislabels, and 351 F1 classes to {args.output}')


if __name__ == '__main__':
    main()

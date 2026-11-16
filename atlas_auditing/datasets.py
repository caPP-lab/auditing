"""Ordered label populations used by the simulator."""
from functools import lru_cache
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

from .strata import stratify_votes

DATA = Path(os.environ.get('ATLAS_DATA_DIR', Path(__file__).resolve().parents[1] / 'data'))
DATASETS = ('kenya', 'cifar10n', 'reimagenet', 'cleanconll')


@lru_cache(maxsize=4)
def load(dataset):
    if dataset == 'cifar10n':
        with np.load(DATA / 'cifar/labels.npz', allow_pickle=False) as saved:
            raw = {k:saved[k] for k in saved.files}
        y, v = np.asarray(raw['random_label1'], int), np.asarray(raw['clean_label'], int)
        rows, row_labels, _ = stratify_votes(y, raw['random_label2'], raw['random_label3'])
        distances = pd.read_csv(DATA / 'cifar/taxonomy_distances.csv', index_col=0).to_numpy(int)
        return dict(y=y, v=v, rows=rows, row_labels=row_labels, categories=10,
                    labels=np.array(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']),
                    distances=distances, d_max=int(distances.max()))
    path = DATA / dataset / 'paired_labels.npz'
    if not path.exists():
        raise FileNotFoundError(f'Prepare {path.name} using data/{dataset}/README.md first.')
    with np.load(path, allow_pickle=False) as saved:
        raw = {k: saved[k].copy() for k in saved.files}
    y, v = raw['submitted'].astype(int), raw['verified'].astype(int)
    labels = raw['labels'] if dataset == 'kenya' else raw['tags']
    row_labels, rows = np.unique(y, return_inverse=True)
    result = dict(y=y, v=v, rows=rows, row_labels=row_labels, labels=labels, categories=len(labels))
    if dataset == 'kenya':
        result.update(rows=raw['record_rows'].astype(int), row_labels=raw['row_labels'].astype(int),
                      strata=raw['strata'].astype(int), stratum_rows=raw['stratum_rows'].astype(int),
                      stratum_labels=raw['stratum_labels'].astype(int))
    else:
        result['target'] = (np.asarray(json.loads((DATA / 'reimagenet/f1_class_ids.json').read_text()), int)
                            if dataset == 'reimagenet' else np.flatnonzero(labels != 'O'))
    return result


def bayesian_population():
    p = load('kenya')
    return dict(y=p['y'], v=p['v'], rows=p['row_labels'], G=len(p['stratum_rows']),
                L=p['categories'], N=len(p['y']), strat_row=p['stratum_labels'],
                strat_rowk=p['stratum_rows'],
                members=[np.flatnonzero(p['strata'] == g) for g in range(len(p['stratum_rows']))])


def truth(dataset, score):
    p = load(dataset)
    y, v = p['y'], p['v']
    if score == 'mislabel':
        return float(np.mean(y != v))
    if score == 'entropy':
        probabilities = np.bincount(v, minlength=p['categories']) / len(v)
        return float(np.exp(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))))
    if score == 'f1':
        a, b = np.isin(y, p['target']), np.isin(v, p['target'])
        return float(2 * np.sum((y == v) & a) / (a.sum() + b.sum()))
    costs = np.eye(10) if score == 'worst3_precision' else p['distances'] / p['d_max']
    means = np.bincount(y, weights=costs[y,v], minlength=10) / np.bincount(y, minlength=10)
    return float(np.sort(means)[:3].mean() if score == 'worst3_precision' else np.sort(means)[-3:].mean())

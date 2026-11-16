from __future__ import annotations
from typing import Sequence
import numpy as np
AGREEMENT_GROUPS = ('both_support', 'one_support', 'both_oppose', 'all_different')
GROUP_BOTH_SUPPORT = 0
GROUP_ONE_SUPPORT = 1
GROUP_BOTH_OPPOSE = 2
GROUP_ALL_DIFFERENT = 3

def agreement_groups(submitted: Sequence[int], auxiliary_one: Sequence[int], auxiliary_two: Sequence[int]) -> np.ndarray:
    submitted = np.asarray(submitted, dtype=int)
    one = np.asarray(auxiliary_one, dtype=int)
    two = np.asarray(auxiliary_two, dtype=int)
    if submitted.ndim != 1 or one.shape != submitted.shape or two.shape != submitted.shape:
        raise ValueError('submitted and auxiliary labels must be aligned vectors')
    groups = np.full(submitted.shape, GROUP_ALL_DIFFERENT, dtype=int)
    both_support = (one == submitted) & (two == submitted)
    both_oppose = (one == two) & (one != submitted)
    one_support = ((one == submitted) | (two == submitted)) & ~both_support
    groups[both_support] = GROUP_BOTH_SUPPORT
    groups[one_support] = GROUP_ONE_SUPPORT
    groups[both_oppose] = GROUP_BOTH_OPPOSE
    return groups

def stratify_votes(submitted: Sequence[int], auxiliary_one: Sequence[int], auxiliary_two: Sequence[int], *, classes: int=10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    submitted = np.asarray(submitted, dtype=int)
    if submitted.min() < 0 or submitted.max() >= classes:
        raise ValueError('submitted labels lie outside the declared vocabulary')
    groups = agreement_groups(submitted, auxiliary_one, auxiliary_two)
    strata = submitted * len(AGREEMENT_GROUPS) + groups
    row_to_submitted = np.repeat(np.arange(classes, dtype=int), len(AGREEMENT_GROUPS))
    return (strata, row_to_submitted, groups)

def _categorical_row_sampler(rng: np.random.Generator, desired: np.ndarray, active: np.ndarray) -> tuple[int, np.ndarray, None]:
    probabilities = np.asarray(desired, dtype=float).copy()
    probabilities[~np.asarray(active, dtype=bool)] = 0.0
    if probabilities.sum() <= 0.0:
        probabilities = np.asarray(active, dtype=float)
    probabilities /= probabilities.sum()
    selected = int(rng.choice(probabilities.size, p=probabilities))
    return (selected, probabilities, None)

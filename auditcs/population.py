"""Fixed finite populations used by the auditing experiments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _largest_remainder_counts(total: int, probabilities: Sequence[float]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    raw = total * probabilities
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


@dataclass(frozen=True)
class AuditPopulation:
    """A completely fixed categorical population.

    ``submitted`` and ``verified`` are integer category codes. ``row_supports``
    are predeclared possible verified destinations for acquisition calculations;
    confidence calculations may still use the full vocabulary.
    """

    submitted: np.ndarray
    verified: np.ndarray
    labels: tuple[str, ...]
    row_labels: tuple[str, ...]
    row_supports: tuple[tuple[int, ...], ...]
    name: str = "population"

    def __post_init__(self) -> None:
        submitted = np.asarray(self.submitted, dtype=int)
        verified = np.asarray(self.verified, dtype=int)
        if submitted.ndim != 1 or verified.ndim != 1 or submitted.size != verified.size:
            raise ValueError("submitted and verified must be equally sized one-dimensional arrays")
        if submitted.size == 0:
            raise ValueError("population must contain at least one observation")
        if submitted.min() < 0 or submitted.max() >= len(self.row_labels):
            raise ValueError("submitted row code outside row_labels")
        if verified.min() < 0 or verified.max() >= len(self.labels):
            raise ValueError("verified category code outside labels")
        if len(self.row_supports) != len(self.row_labels):
            raise ValueError("one row support is required for every submitted-label row")
        object.__setattr__(self, "submitted", submitted)
        object.__setattr__(self, "verified", verified)

    @property
    def n(self) -> int:
        return int(self.submitted.size)

    @property
    def n_rows(self) -> int:
        return len(self.row_labels)

    @property
    def n_categories(self) -> int:
        return len(self.labels)

    @cached_property
    def row_indices(self) -> tuple[np.ndarray, ...]:
        return tuple(np.flatnonzero(self.submitted == r) for r in range(self.n_rows))

    @cached_property
    def row_sizes(self) -> np.ndarray:
        return np.bincount(self.submitted, minlength=self.n_rows).astype(int)

    @cached_property
    def row_weights(self) -> np.ndarray:
        return self.row_sizes / self.n

    @cached_property
    def true_confusion(self) -> np.ndarray:
        matrix = np.zeros((self.n_rows, self.n_categories), dtype=int)
        np.add.at(matrix, (self.submitted, self.verified), 1)
        return matrix

    @cached_property
    def true_composition(self) -> np.ndarray:
        return np.bincount(self.verified, minlength=self.n_categories) / self.n

    def random_orders(self, seed: int) -> tuple[np.ndarray, ...]:
        rng = np.random.default_rng(seed)
        return tuple(rng.permutation(indices) for indices in self.row_indices)

    @classmethod
    def from_labels(
        cls,
        submitted_labels: Iterable[str],
        verified_labels: Iterable[str],
        *,
        name: str = "population",
        declared_supports: Mapping[str, Iterable[str]] | None = None,
    ) -> "AuditPopulation":
        submitted_text = np.asarray(list(submitted_labels), dtype=str)
        verified_text = np.asarray(list(verified_labels), dtype=str)
        if submitted_text.size != verified_text.size:
            raise ValueError("submitted and verified labels must have equal length")

        labels = tuple(sorted(set(submitted_text) | set(verified_text)))
        row_labels = tuple(sorted(set(submitted_text)))
        label_to_code = {label: i for i, label in enumerate(labels)}
        row_to_code = {label: i for i, label in enumerate(row_labels)}
        submitted = np.fromiter((row_to_code[x] for x in submitted_text), dtype=int)
        verified = np.fromiter((label_to_code[x] for x in verified_text), dtype=int)

        if declared_supports is None:
            full = tuple(range(len(labels)))
            row_supports = tuple(full for _ in row_labels)
        else:
            supports = []
            for row_label in row_labels:
                values = set(declared_supports.get(row_label, labels))
                values.add(row_label)
                unknown = values.difference(label_to_code)
                if unknown:
                    raise ValueError(f"unknown support labels for {row_label}: {sorted(unknown)}")
                supports.append(tuple(sorted(label_to_code[x] for x in values)))
            row_supports = tuple(supports)

        return cls(
            submitted=submitted,
            verified=verified,
            labels=labels,
            row_labels=row_labels,
            row_supports=row_supports,
            name=name,
        )


def load_kenya_population(path: str | Path) -> AuditPopulation:
    """Load the requested row-level replay population from ``labels.csv``."""

    frame = pd.read_csv(path, usecols=["original_label", "updated_label"])
    if frame.isna().any().any():
        raise ValueError("original_label and updated_label must be complete")
    return AuditPopulation.from_labels(
        frame["original_label"].astype(str),
        frame["updated_label"].astype(str),
        name="Kenya biodiversity row-level replay",
    )


def make_synthetic_population(
    *,
    seed: int = 20260822,
    scale: float = 1.0,
    full_support: bool = False,
) -> AuditPopulation:
    """Construct a heterogeneous fixed population with directional confusions.

    The smallest submitted rows have high-impact destination supports, creating a
    useful test of downstream priority. Counts are fixed before audit; ``seed``
    changes only the record ordering inside the constructed population.
    """

    labels = ("Common_A", "Common_B", "Medium_A", "Medium_B", "Rare_A", "Rare_B")
    base_sizes = np.array([1200, 900, 700, 500, 300, 200])
    row_sizes = np.maximum(20, np.rint(scale * base_sizes).astype(int))
    probabilities = np.array(
        [
            [0.94, 0.05, 0.01, 0.00, 0.00, 0.00],
            [0.14, 0.80, 0.06, 0.00, 0.00, 0.00],
            [0.00, 0.08, 0.79, 0.11, 0.02, 0.00],
            [0.09, 0.00, 0.12, 0.65, 0.12, 0.02],
            [0.22, 0.00, 0.08, 0.00, 0.55, 0.15],
            [0.30, 0.00, 0.15, 0.00, 0.20, 0.35],
        ]
    )
    rng = np.random.default_rng(seed)
    submitted_parts: list[np.ndarray] = []
    verified_parts: list[np.ndarray] = []
    supports: list[tuple[int, ...]] = []

    for r, size in enumerate(row_sizes):
        counts = _largest_remainder_counts(int(size), probabilities[r])
        verified = np.repeat(np.arange(len(labels)), counts)
        verified = rng.permutation(verified)
        submitted_parts.append(np.full(size, r, dtype=int))
        verified_parts.append(verified)
        if full_support:
            supports.append(tuple(range(len(labels))))
        else:
            supports.append(tuple(np.flatnonzero(probabilities[r] > 0)))

    return AuditPopulation(
        submitted=np.concatenate(submitted_parts),
        verified=np.concatenate(verified_parts),
        labels=labels,
        row_labels=labels,
        row_supports=tuple(supports),
        name="heterogeneous synthetic population",
    )

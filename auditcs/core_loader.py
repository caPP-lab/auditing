"""Load the numerical entropy routines and optional private population files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE_ROOT = PROJECT_ROOT / "data" / "private_kenya"
DEFAULT_REFERENCE_ROOT = DEFAULT_PRIVATE_ROOT
CORE_ROOT = Path(__file__).resolve().parent / "entropy_core"


@dataclass(frozen=True)
class PrivateInputVerification:
    root: str
    population_sha256: str


_LOADED: dict[str, ModuleType] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_reference_bundle(
    root: Path = DEFAULT_REFERENCE_ROOT,
    *,
    verify_all: bool = True,
) -> PrivateInputVerification:
    """Verify that the private replay population required by Kenya runs exists."""

    del verify_all
    root = Path(root).resolve()
    population = root / "replay_population.npz"
    if not population.is_file():
        raise FileNotFoundError(
            "The Kenya record-level data are confidential and are not included. "
            f"Expected a private replay population at {population}."
        )
    return PrivateInputVerification(
        root=str(root),
        population_sha256=_sha256(population),
    )


def _load_reference_modules(
    root: Path = DEFAULT_REFERENCE_ROOT,
) -> dict[str, ModuleType]:
    """Load the bundled numerical routines used by the entropy certificate."""

    del root
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    sys.path.insert(0, str(CORE_ROOT))
    try:
        names = (
            "convex_audit_online",
            "convex_profile_replay",
            "directional_observations",
            "metadata_prediction_research",
            "smallpilot_research",
        )
        _LOADED = {name: importlib.import_module(name) for name in names}
    finally:
        if sys.path and sys.path[0] == str(CORE_ROOT):
            sys.path.pop(0)
    return _LOADED


def load_reference_population(
    root: Path = DEFAULT_REFERENCE_ROOT,
) -> dict[str, np.ndarray]:
    """Load the private replay population when supplied by an authorized user."""

    verify_reference_bundle(root)
    archive = np.load(Path(root) / "replay_population.npz", allow_pickle=False)
    return {name: archive[name].copy() for name in archive.files}


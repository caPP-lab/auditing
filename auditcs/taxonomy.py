"""Frozen taxonomy artifacts for canonical label-audit experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import numpy as np


WORDNET_VERSION = "Princeton WordNet 3.0"
WORDNET_ARCHIVE_URL = (
    "https://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz"
)
WORDNET_ARCHIVE_SHA256 = (
    "640db279c949a88f61f851dd54ebbb22d003f8b90b85267042ef85a3781d3a52"
)

CIFAR10_WORDNET_SENSES = (
    (0, "airplane", "airplane%1:06:00::", "fixed-wing powered aircraft"),
    (1, "automobile", "automobile%1:06:00::", "four-wheel motor vehicle"),
    (2, "bird", "bird%1:05:00::", "feathered egg-laying vertebrate"),
    (3, "cat", "cat%1:05:00::", "feline mammal, including domestic cats"),
    (4, "deer", "deer%1:05:00::", "cervid mammal"),
    (5, "dog", "dog%1:05:00::", "domesticated member of genus Canis"),
    (6, "frog", "frog%1:05:00::", "tailless amphibian"),
    (7, "horse", "horse%1:05:00::", "domesticated equine mammal"),
    (8, "ship", "ship%1:06:00::", "vessel carrying passengers or freight"),
    (9, "truck", "truck%1:06:00::", "automotive hauling vehicle"),
)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class NounSynset:
    offset: int
    lexicographer_file: int
    lemmas: tuple[str, ...]
    hypernyms: tuple[int, ...]
    gloss: str

    @property
    def synset_id(self) -> str:
        return f"{self.offset:08d}-n"


def parse_wordnet_nouns(path: str | Path) -> dict[int, NounSynset]:
    """Parse the noun synsets and hypernym edges from a WordNet data file."""

    synsets: dict[int, NounSynset] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line[:1].isdigit():
                continue
            body, _, gloss = raw_line.rstrip("\n").partition(" | ")
            fields = body.split()
            offset = int(fields[0])
            lexicographer_file = int(fields[1])
            if fields[2] != "n":
                continue
            word_count = int(fields[3], 16)
            cursor = 4
            lemmas = tuple(
                fields[cursor + 2 * index].replace("_", " ")
                for index in range(word_count)
            )
            cursor += 2 * word_count
            pointer_count = int(fields[cursor])
            cursor += 1
            hypernyms: list[int] = []
            for _ in range(pointer_count):
                symbol, target, target_pos, _source_target = fields[cursor : cursor + 4]
                cursor += 4
                if symbol in {"@", "@i"} and target_pos == "n":
                    hypernyms.append(int(target))
            synsets[offset] = NounSynset(
                offset=offset,
                lexicographer_file=lexicographer_file,
                lemmas=lemmas,
                hypernyms=tuple(sorted(set(hypernyms))),
                gloss=gloss,
            )
    if not synsets:
        raise ValueError("WordNet noun data contained no synsets")
    unknown = {
        parent
        for synset in synsets.values()
        for parent in synset.hypernyms
        if parent not in synsets
    }
    if unknown:
        raise ValueError(f"WordNet noun hypernyms reference missing synsets: {unknown}")
    return synsets


def parse_sense_index(path: str | Path) -> dict[str, int]:
    senses: dict[str, int] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2 and "%1:" in fields[0]:
                senses[fields[0]] = int(fields[1])
    if not senses:
        raise ValueError("WordNet sense index contained no noun senses")
    return senses


def _all_hypernym_paths(
    offset: int,
    synsets: dict[int, NounSynset],
    memo: dict[int, tuple[tuple[int, ...], ...]],
    active: frozenset[int] = frozenset(),
) -> tuple[tuple[int, ...], ...]:
    if offset in memo:
        return memo[offset]
    if offset in active:
        raise ValueError("cycle detected in WordNet hypernym graph")
    parents = synsets[offset].hypernyms
    if not parents:
        paths = ((offset,),)
    else:
        paths = tuple(
            (offset, *tail)
            for parent in parents
            for tail in _all_hypernym_paths(
                parent, synsets, memo, active | {offset}
            )
        )
    memo[offset] = tuple(sorted(set(paths)))
    return memo[offset]


def _undirected_hypernym_adjacency(
    synsets: dict[int, NounSynset],
) -> dict[int, tuple[int, ...]]:
    adjacency: dict[int, set[int]] = {offset: set() for offset in synsets}
    for offset, synset in synsets.items():
        for parent in synset.hypernyms:
            adjacency[offset].add(parent)
            adjacency[parent].add(offset)
    return {offset: tuple(sorted(neighbors)) for offset, neighbors in adjacency.items()}


def _shortest_path_length(
    source: int,
    target: int,
    adjacency: dict[int, tuple[int, ...]],
) -> int:
    if source == target:
        return 0
    queue: deque[tuple[int, int]] = deque([(source, 0)])
    seen = {source}
    while queue:
        node, distance = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor == target:
                return distance + 1
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    raise ValueError(f"no noun-hypernym path connects {source} and {target}")


def _serialized_path(
    offsets: Iterable[int], synsets: dict[int, NounSynset]
) -> list[dict[str, object]]:
    return [
        {
            "synset_id": synsets[offset].synset_id,
            "lemmas": list(synsets[offset].lemmas),
        }
        for offset in offsets
    ]


def build_cifar10_wordnet_artifact(
    wordnet_root: str | Path,
) -> dict[str, object]:
    root = Path(wordnet_root)
    data_path = root / "dict" / "data.noun"
    sense_path = root / "dict" / "index.sense"
    if not data_path.is_file() or not sense_path.is_file():
        raise FileNotFoundError("WordNet root must contain dict/data.noun and dict/index.sense")
    synsets = parse_wordnet_nouns(data_path)
    sense_index = parse_sense_index(sense_path)
    selected_offsets: list[int] = []
    classes: list[dict[str, object]] = []
    path_memo: dict[int, tuple[tuple[int, ...], ...]] = {}
    for class_id, label, sense_key, rationale in CIFAR10_WORDNET_SENSES:
        if sense_key not in sense_index:
            raise KeyError(f"frozen WordNet sense is absent: {sense_key}")
        offset = sense_index[sense_key]
        synset = synsets[offset]
        lemma_from_key = sense_key.split("%", 1)[0].replace("_", " ")
        if lemma_from_key not in synset.lemmas:
            raise ValueError(f"sense key {sense_key} does not match synset lemmas")
        paths = _all_hypernym_paths(offset, synsets, path_memo)
        selected_offsets.append(offset)
        classes.append(
            {
                "class_id": class_id,
                "released_label": label,
                "wordnet_version": WORDNET_VERSION,
                "synset_id": synset.synset_id,
                "sense_key": sense_key,
                "lemmas": list(synset.lemmas),
                "gloss": synset.gloss,
                "sense_selection_rationale": rationale,
                "hypernym_paths": [
                    _serialized_path(path, synsets) for path in paths
                ],
                "ambiguity_note": (
                    "The released flat class name is broader than some images; "
                    "the frozen first concrete noun sense is used without consulting "
                    "the confusion matrix."
                ),
            }
        )

    adjacency = _undirected_hypernym_adjacency(synsets)
    dimension = len(selected_offsets)
    distances = np.zeros((dimension, dimension), dtype=int)
    for left in range(dimension):
        for right in range(left + 1, dimension):
            distance = _shortest_path_length(
                selected_offsets[left], selected_offsets[right], adjacency
            )
            distances[left, right] = distance
            distances[right, left] = distance
    d_max = int(distances.max())
    if d_max <= 0:
        raise ValueError("CIFAR WordNet distance matrix is degenerate")

    neighborhoods: list[list[int]] = []
    for row in range(dimension):
        positive = np.sort(distances[row][distances[row] > 0])
        if positive.size < 2:
            raise ValueError("taxonomy has fewer than two distinct neighboring labels")
        cutoff = int(positive[1])
        neighborhood = np.flatnonzero(
            (distances[row] == 0) | (distances[row] <= cutoff)
        )
        neighborhoods.append([int(value) for value in neighborhood])

    matrix = distances.tolist()
    return {
        "schema_version": "cifar10-wordnet-3.0-v1",
        "dataset": "CIFAR-10",
        "taxonomy": WORDNET_VERSION,
        "source": {
            "archive_url": WORDNET_ARCHIVE_URL,
            "archive_sha256": WORDNET_ARCHIVE_SHA256,
            "data_noun_sha256": sha256_file(data_path),
            "index_sense_sha256": sha256_file(sense_path),
        },
        "sense_selection_rule": (
            "Frozen first concrete noun sense matching each released CIFAR-10 label; "
            "selected without inspecting audit outcomes."
        ),
        "distance_rule": (
            "Unweighted shortest-path length in the undirected graph induced by "
            "Princeton WordNet 3.0 noun hypernym and instance-hypernym edges."
        ),
        "classes": classes,
        "integer_distance_matrix": matrix,
        "integer_distance_matrix_sha256": sha256_json(matrix),
        "d_max": d_max,
        "normalized_distance_rule": "integer_distance / d_max",
        "neighborhood_rule": (
            "self plus the two nearest distinct labels, including every label tied "
            "at the second-neighbor WordNet-distance cutoff"
        ),
        "neighborhoods": neighborhoods,
    }


def load_cifar10_taxonomy(path: str | Path) -> dict[str, object]:
    artifact = json.loads(Path(path).read_text())
    matrix = np.asarray(artifact["integer_distance_matrix"], dtype=int)
    if matrix.shape != (10, 10):
        raise ValueError("CIFAR-10 taxonomy must contain a 10 by 10 matrix")
    if not np.array_equal(matrix, matrix.T) or np.any(np.diag(matrix) != 0):
        raise ValueError("taxonomy distance matrix must be symmetric with zero diagonal")
    if np.any(matrix < 0) or int(matrix.max()) != int(artifact["d_max"]):
        raise ValueError("taxonomy distance values or d_max are invalid")
    if sha256_json(matrix.tolist()) != artifact["integer_distance_matrix_sha256"]:
        raise ValueError("taxonomy distance matrix hash mismatch")
    class_ids = [int(item["class_id"]) for item in artifact["classes"]]
    if class_ids != list(range(10)):
        raise ValueError("taxonomy classes must cover CIFAR-10 in released order")
    return artifact


__all__ = [
    "CIFAR10_WORDNET_SENSES",
    "WORDNET_ARCHIVE_SHA256",
    "WORDNET_ARCHIVE_URL",
    "WORDNET_VERSION",
    "build_cifar10_wordnet_artifact",
    "load_cifar10_taxonomy",
    "parse_sense_index",
    "parse_wordnet_nouns",
    "sha256_file",
    "sha256_json",
]

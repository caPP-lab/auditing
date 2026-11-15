# Supplementary code

This repository contains the experiment and figure-generation code for the paper. It includes public audit inputs, data-preparation instructions, the synthetic experiment, and species-named aggregate counts for the Kenyan biodiversity experiments.

The record-level Kenyan biodiversity data remain confidential. The preprocessed confusion matrix is available in `data/kenya`, with a matching copy and the other aggregate values used in the paper under `results/figure_data`.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Reproduce the public experiments

### Data

- [`data/kenya/confusion_matrix.csv`](data/kenya/confusion_matrix.csv): the preprocessed Kenya biodiversity confusion matrix with species/category names (33,403 records); see [provenance and preprocessing](#kenyan-biodiversity-data) below.
- `data/cifar/`: the CIFAR-10N input used in the experiments.
- [`data/cleanconll/`](data/cleanconll/README.md): the exact, ordered labels-only CleanCoNLL audit population (301,353 tokens), with provenance and score definitions. No corpus text is included.
- [`data/reimagenet/`](data/reimagenet/README.md): pinned download instructions and a preprocessor for the exact ReImageNet audit population (30,908 images), including the fixed class list for F1. ReImageNet annotations must be obtained separately from their authors.

These additions provide the frozen inputs and preprocessing definitions. The existing experiment scripts below are unchanged; runners for the ReImageNet and CleanCoNLL experiments are not yet included in this repository.

```bash
python scripts/run_cifar_mislabel_study.py all --workers 4
python scripts/compute_mislabel_comparison_baselines.py
python scripts/report_cifar_mislabel_figure.py

python scripts/run_cifar_tail_betting_study.py all --workers 4 \
  --output results/cifar_worst_three \
  --seeds 2026100300,2026100301,2026100302,2026100303,2026100304,2026100305,2026100306,2026100307,2026100308,2026100309 \
  --pas-variant robust70
python scripts/report_cifar_tail_betting_figure.py

python scripts/run_synthetic_linear_pas_study.py all --workers 4
python scripts/report_synthetic_linear_pas_figures.py
```

The reporting scripts write regenerated figures to `figures/`. Saved experiment outputs are under `results/`.

### Kenyan biodiversity data

The data come from camera-trap biodiversity monitoring in Kenya, supplied by a conservation NGO, and pair submitted identifications with expert-reviewed labels. Preprocessing uses an explicit keyword-based mapping to merge spelling, capitalization, and subtype variants into shared categories (for example, zebra variants to `Zebra` and gazelle variants to `Gazelle`). All 33,403 records are retained across 50 categories, including `Blank`, `Human`, and `Vehicle`.

In [`data/kenya/confusion_matrix.csv`](data/kenya/confusion_matrix.csv), rows are submitted labels, columns are expert-reviewed labels, and cells contain integer record counts. Both axes use the same category order; 46 rows are nonempty. The matrix has 6,002 off-diagonal records (17.97% label disagreement) and exactly preserves the preprocessed experiment counts. The existing [`results/figure_data/kenya_compact_confusion_matrix.csv`](results/figure_data/kenya_compact_confusion_matrix.csv) now contains the same named matrix.

The release contains aggregate counts, without images, locations, original record order, or camera-station assignments. The matrix therefore does not reproduce the exact seeded record sequences or station-stratified experiments on its own.

## Bayesian experiment

The Bayesian implementation is `scripts/run_bayesian_diversity_study.py`. It requires the confidential record-level Kenya CSV and therefore cannot be rerun from the public supplement alone. The per-path output is preserved at `results/kenya_bayesian/results.jsonl`.


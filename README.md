# Supplementary code

This repository contains the experiment and figure-generation code for the paper. It includes the CIFAR-10N input used in the experiments, the synthetic experiment, anonymized aggregate results for the Kenyan biodiversity experiments.

The record-level Kenyan biodiversity data are confidential and are not distributed. The corresponding scripts are included for methodological inspection, and the aggregate values used in the paper are under `results/figure_data`.

## Setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Reproduce the public experiments

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

## Bayesian experiment

The Bayesian implementation is `scripts/run_bayesian_diversity_study.py`. It requires the confidential record-level Kenya CSV and therefore cannot be rerun from the public supplement alone. The per-path output is preserved at `results/kenya_bayesian/results.jsonl`.


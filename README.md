# ATLAS: categorical-label auditing

Code, data, results, and experimental figures for the camera-ready paper. The
release contains the reported experiments with their frozen settings and seeds.

## Setup

Python 3.12 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

Kenya, CIFAR-10N, and CleanCoNLL inputs are included. Before rerunning ReImageNet,
follow [its pinned download and preprocessing instructions](data/reimagenet/README.md).
Downloading images is unnecessary. The saved results can be summarized and plotted
without downloading ReImageNet.

## Reproduce tables and figures

From the release root:

```bash
python report_results.py
python make_figures.py
```

These commands regenerate `results/*summary.csv`, `results/tables/*.tex`, and
`figures/*.pdf` from the included per-path results. LaTeX tables require
`booktabs` and `graphicx`.

## Rerun experiments

```bash
# One frozen seed across the seven plotted Kenya mislabel methods:
python run_experiments.py --study mislabel --datasets kenya --seed-index 0

# All reported experiments and seeds:
python run_experiments.py --study all --workers 4
python report_results.py --runs runs --output reproduced
python make_figures.py --results reproduced --output reproduced/figures
```

The runner saves each completed path separately and skips existing completed
files when restarted. `--study` also accepts `nonlinear`, `bayesian`, `synthetic`,
`epsilon`, `initialization`, and `misspecification`. `--datasets` selects among
`kenya`, `cifar10n`, `reimagenet`, and `cleanconll`. The complete-run collector
expects every frozen path. `--seed-index` is for individual-path reproduction.

| Study | Paths per policy/setting | Audit budget | Reported updates |
| --- | ---: | ---: | --- |
| Mislabel rate, four datasets | 100 | 5,000 | Every audit; target checks from 100 |
| Kenya Hill–Shannon diversity, ATLAS and uncertainty-only | 100 | 3,000 | Every 25 audits |
| Other nonlinear runs, including Kenya PPR/MR | 100 | 10,000 | Every 25 audits |
| Kenya Bayesian diversity | 100 | 1,000 | From 100, every 25; stop at target |
| Synthetic, 20 rows | 100 | 800 | Every 25 audits |
| Synthetic scaling, 8/12/20/32/48 rows | 5 | 800 | Every 25 audits |
| Exploration weight and predictor misspecification | 100 | 5,000 | Every audit |
| Kenya initialization, linear/nonlinear | 100 | 5,000 / 3,000 | Every audit / every 25 |

Settings are in `settings.json`; executable definitions and seeds are in
`atlas_auditing/experiments.py`. Seed pairs in the settings file give inclusive
first and last seed values. No configuration search is performed by the runner.
Chebyshev sample counts are computed analytically.

## Contents

- `data/`: ordered label inputs, taxonomy distances, and preprocessing documentation.
- `atlas_auditing/`: inference, sampling, numerical bounds, and reporting code.
- `results/`: current per-path outputs and derived summaries; [file definitions](results/README.md).
- `figures/`: only experimental figures used in the main text and appendix;
  [paper-to-file mapping](figures/README.md).
- `tests/`: data, reporting, and numerical replay checks.
- `SHA256SUMS`: checksums of release files.

Mislabel and nonlinear curves plot median first-attainment times as a function
of precision. Their target intersections and table entries use the same stopping
times. Bands give the horizontal 25th–75th percentile attainment-time range.
Unreached targets remain right-censored; all paths enter the summaries. Nonlinear
stopping times use observed checkpoints, with no interpolation within a path.
The synthetic figure reports interval widths and paired terminal-width reductions.

## Baseline sources

The linear comparisons implement the two reported risk-limiting financial
auditing variants of Shekhar et al. (2023), the finite-population
prior–posterior-ratio confidence sequence of Waudby-Smith and Ramdas (2020),
and the stated fixed-sample bounds. Nonlinear PPR projects simultaneous
component intervals onto each score. MR stitching instantiates the
reverse-submartingale framework of Manole and Ramdas (2023), with
finite-population bounds and score-specific inversion. These adaptations and
their sampling rules are described in the paper's appendix. Uncertainty-only
uses the ATLAS intervals and removes downstream-score influence from allocation.

Dataset provenance and upstream terms are recorded in each dataset's README.

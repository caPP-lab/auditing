# Saved results

All real-data experiment cells contain 100 paths. Seeds and budgets are defined
in `settings.json` at the release root. The synthetic scaling comparison uses
five paired paths per row count, as reported in the appendix.

- `mislabel/<dataset>/<method>.npz`: seeds, audit counts, per-path lower/upper
  endpoints, reported half-widths, and empty-interval indicators. Empty intervals
  cannot attain a precision target. The CleanCoNLL Wald and Serfling half-widths
  are the unclipped reference widths used in the paper.
- `nonlinear/checkpoints.csv.gz`: score, method, seed, audit count, endpoints,
  truth, empty indicator, and coverage at each 25-audit checkpoint.
  Reporting adds the deterministic pre-audit F1 interval `[0,1]` at audit zero.
- `bayesian/paths.jsonl`: stopping audit, posterior endpoints, coverage at
  stopping, whole-path coverage through stopping, and evaluated checkpoints.
- `synthetic/`: the fixed population and checkpoint outputs for the main and
  scaling experiments.
- `sensitivity/*_paths.csv`: first-target times and whole-path coverage for each
  frozen seed and sensitivity setting. Whole-path checks include every published
  interval update through the budget.
- `confusion/`: the displayed count matrices. The ReImageNet display uses the
  13 feline classes and an `Other` destination column; auditing uses all classes.
- `linear_curves.csv.gz`, `nonlinear_curves.csv.gz`: median and quartile
  first-attainment times for a precision grid containing each exact target.
- `target_summary.csv`, `per_path_target_times.csv.gz`: the table statistics
  and corresponding per-path stopping times. Infinity denotes nonattainment.
- `*_summary.csv`, `sensitivity/*_summary.csv`, `nonlinear_coverage.csv`: derived
  experiment summaries.
- `tables/`: generated LaTeX result tables.

`python report_results.py` regenerates summaries and tables from these inputs.
Raw floating-point endpoints and stopping times are preserved. LaTeX audit
counts are rounded to the nearest integer; censored paths remain in quantiles.

# CIFAR-10N labels

Source: Wei et al. (2022), *Learning with Noisy Labels Revisited: A Study Using
Real-World Human Annotations*, [CIFAR-N](https://github.com/UCSC-REAL/cifar-10-100n).
Upstream terms are in `LICENSE.txt`.

`labels.npz` contains the four annotation arrays used here, extracted without
reordering from the official `CIFAR-10_human_ordered.npy` dictionary:
`clean_label`, `random_label1`, `random_label2`, and `random_label3`.
All 50,000 CIFAR-10 training images are retained. `random_label1` is the
submitted label and `clean_label` is the audit reference: 8,617 disagreements.
No images are needed for the experiments.

For mislabel-rate experiments, `random_label2` and `random_label3` are pre-audit
covariates. Their agreement pattern with the submitted label defines four
groups per submitted class, for 40 strata. Predictions start at error
probability 0.5 and update from audited records. The nonlinear experiments use
the ten submitted classes as strata.

`taxonomy_distances.csv` contains the fixed WordNet 3.0 shortest-path distances
between the ten class concepts, in the same class order as the label codes.
Worst-three taxonomic risk divides these costs by the maximum
distance, 15. Worst-three precision averages the three smallest submitted-class
precision values. The runner evaluates both scores on the complete population.

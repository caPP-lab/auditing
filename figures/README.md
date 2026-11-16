# Experimental figures

Run `python make_figures.py` from the release root to regenerate these PDFs.

| Paper figure | PDF file |
| --- | --- |
| 2, Kenya | `kenya_confusion_matrix_panel.pdf` and `mislabel_kenya.pdf` |
| 3, four nonlinear settings | `nonlinear_four_datasets_compact_row.pdf` |
| 4, CIFAR-10N mislabel rate | `mislabel_cifar_with_confusion.pdf` |
| 5, worst-three taxonomic risk | `nonlinear_cifar_worst3_taxonomic_risk_appendix.pdf` |
| 6, ReImageNet mislabel rate | `mislabel_reimagenet_with_confusion.pdf` |
| 7, CleanCoNLL mislabel rate | `mislabel_cleanconll_with_confusion.pdf` |
| 8, synthetic experiment | `synthetic.pdf` |
| 9, exploration-weight sensitivity | `epsilon_all_datasets.pdf` |
| 10, predictive initialization | `kenya_prior_sensitivity_audits_coverage_multiplot.pdf` |

The uncertainty-only ablation is included in the nonlinear result table. The
main nonlinear figure and standalone taxonomic-risk figure show the four
primary methods. All mislabel plots include both reported RLFA variants.

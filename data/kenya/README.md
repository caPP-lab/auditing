# Kenya biodiversity labels

Camera-trap observations from biodiversity monitoring in Kenya, supplied by a
conservation NGO, pair submitted identifications with expert-reviewed labels.
An explicit keyword mapping merges spelling, capitalization, and subtype
variants into common categories (for example, zebra variants into `Zebra` and
gazelle variants into `Gazelle`). All 33,403 records are retained in 50
categories, including `Blank`, `Human`, and `Vehicle`; 6,002 labels disagree.

`confusion_matrix.csv` contains integer counts, with submitted labels in rows
and expert labels in columns. Both axes have the same named-category order;
46 submitted rows are nonempty.

`paired_labels.npz` contains the exact ordered, preprocessed label population
used for seeded simulations. It contains no images, coordinates, timestamps,
original record identifiers, or camera-station names. Load with
`np.load(path, allow_pickle=False)`:

| Array | Definition |
| --- | --- |
| `submitted`, `verified` | Ordered integer category codes, 33,403 records |
| `labels` | Category names indexed by code |
| `record_rows` | Nonempty submitted-label row for each record |
| `row_labels` | Category code of each of the 46 rows |
| `strata` | Numeric submitted-label × camera-station group for each record |
| `stratum_rows`, `stratum_labels` | Submitted row and category code for each of 300 groups |

Mislabel experiments sample within submitted-label rows. Diversity and Bayesian
experiments use the numeric station-by-label strata. Group memberships and record
order are retained for exact seeded replay; station names are omitted. Do not
sort or shuffle these arrays before running the supplied experiments.

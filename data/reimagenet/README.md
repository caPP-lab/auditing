# ReImageNet audit population

ReImageNet reannotates the 50,000 ImageNet-1K validation images. Obtain the
annotations from [vrg-prague/ReImageNet](https://huggingface.co/datasets/vrg-prague/ReImageNet),
accepting the authors' access terms. We do not redistribute the annotations or images.
The dataset card lists CC BY 4.0 and also requires author permission for redistribution;
follow the upstream terms and cite [*Doomed to Re-Annotate, Forever: The ImageNet
Story*](https://arxiv.org/abs/2608.13783).

## Reproduce the exact input

1. Download **`reannotation.jsonl`** at revision
   **`b276020137db716a492c7a0af6fd05cb63be6599`**:
   [pinned file](https://huggingface.co/datasets/vrg-prague/ReImageNet/blob/b276020137db716a492c7a0af6fd05cb63be6599/reannotation.jsonl).
   If access is gated, sign in/request access through the authors' page. Keep the
   original bytes and line ordering. Images, bounding boxes, and model predictions
   are not needed for these label-auditing experiments.
2. After the repository setup, run from its root:

   ```bash
   python data/reimagenet/preprocess.py \
     --input /path/to/reannotation.jsonl \
     --output data/reimagenet/paired_labels.npz
   ```

The script uses only NumPy and the Python standard library. It refuses a different
source SHA-256, validates the ordered output against the experiment inputs, and
does not overwrite an existing output. Raw-file SHA-256:

```text
e25897e79ce5bf0e47a1593e702b75cba70b8409861f93e94af6f47f7d7f91ea
```

## Exact preprocessing and scores

- Retain a record only when **both** `original_class` and `reannotated_labels`
  contain exactly one class ID, with both IDs in `0..999`.
- Submitted label = `original_class[0]`; verified label = `reannotated_labels[0]`.
  Exclude empty labels, multi-label records, and invalid/out-of-distribution IDs.
  Do not deduplicate multi-label sets into singletons, merge equivalent classes,
  apply `class_update_config.json`, or remap the original 1,000-class vocabulary.
- Preserve raw JSONL order. Use `image_name` as the record ID. Do not shuffle,
  subsample, split, or filter by correctness or F1 membership. One retained image
  is one audit. Strata are defined by the submitted class only.
- Mislabel rate is the fraction with unequal submitted and verified IDs.
- F1 is exact-class micro-F1 over the **351 fixed human-made/artifact classes**
  in `f1_class_ids.json`. This is our experiment-specific class selection, not an
  official ImageNet superclass or an additional population filter. All 30,908
  records remain auditable, including classes outside that list. For class set
  `S`, submitted ID `y`, and verified ID `v`, the score is
  `2 * sum((y == v) & (y in S)) / (sum(y in S) + sum(v in S))`.
  Cross-boundary errors and confusions between distinct selected classes affect
  F1; correct records outside `S` contribute zero to both sums.

Expected checks: **30,908 records; 2,501 mislabels (8.091756%); 964 submitted
classes; 958 verified classes; F1 = 0.9293055123014637**. Submitted and verified
membership totals for the F1 class set are 9,584 and 9,682 respectively.
These targets describe the retained population, not the entire ImageNet validation set.

## Output

Load with `np.load(path, allow_pickle=False)`. Arrays are:

| Key | Meaning |
| --- | --- |
| `submitted`, `verified` | Ordered integer class IDs, shape `(30908,)` |
| `record_id` | Original image names in the same order |
| `tags` | String IDs `"0"` through `"999"`, indexed by label ID |
| `f1_class_ids` | Sorted fixed set of 351 class IDs |

The ordered two-column `(submitted, verified)` array, serialized as little-endian
64-bit integers in row-major order, has SHA-256
`b31cfdf49f8bbbd361fe7742d3acb243a06a68659e6ed1721b460ae8857a4ced`.
This array checksum is independent of ZIP/container metadata in the NPZ file.
Record order matters for seed-level replay. The preprocessor reproduces inputs;
it does not run an auditing policy. Experiment runners will be added separately.

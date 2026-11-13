# CleanCoNLL audit population

`paired_labels.npz` is the exact labels-only input used in our CleanCoNLL mislabel-rate,
F1, and exploration-weight experiments. It contains **301,353 aligned tokens**
pooled in train, dev, test order, with **3,612 label disagreements (1.198594%)**.
No token text, images, Wikipedia links, or other corpus content is included.

## Provenance and preprocessing

Source: [CleanCoNLL](https://github.com/flairNLP/CleanCoNLL/tree/bedc569f0983ddd39c2c729fb5df64be710bcab4),
commit `bedc569f0983ddd39c2c729fb5df64be710bcab4`.
Cite Rücker and Akbik (2023), *CleanCoNLL: A Nearly Noise-Free Named Entity
Recognition Dataset*, [EMNLP](https://doi.org/10.18653/v1/2023.emnlp-main.533),
and the [CoNLL-2003 shared task](https://aclanthology.org/W03-0419/).
The annotations are derived from those datasets and remain subject to upstream
terms; this repository does not grant additional rights to the original corpus.
The upstream authors distribute masked annotations separately from licensed token text.

- Start from the original CoNLL-2003 labels and apply the official CleanCoNLL
  token patch to align the two versions. The original archive used was
  `https://data.deepai.org/conll2003.zip`, SHA-256
  `96a104d174ddae7558bab603f19382c5fe02ff1da5c077a7f3ce2ced1578a2c3`.
- Retain exact unchanged token lines outside patch edits. Exclude inserted,
  removed, and replaced tokens, blank sentence separators, and `-DOCSTART-` markers.
  Do not realign repeated token strings by approximate matching.
- Submitted labels come from the original corpus; verified labels come from
  the **last column** of `cleanconll_annotations.{train,dev,test}`, the final
  CleanCoNLL annotation, not CleanCoNLL* or Wikipedia/NEL labels.
- Normalize both sequences to BIO2 before selecting aligned tokens: an `I-X`
  becomes `B-X` when it begins a sentence or follows `O` or a different entity
  type. Preserve the other tags. Reset at blank lines and document markers.
- Retain **all `O` tokens**. Each retained token is one audit, stratified only by
  its submitted BIO tag. There is no sentence-level sampling or train/test holdout.
- Preserve corrected-file line order within train, dev, test. Retained split
  sizes are 203,603 / 51,350 / 46,400, with 2,088 / 694 / 830 disagreements.

## Loading and score definitions

```python
import numpy as np

with np.load('data/cleanconll/paired_labels.npz', allow_pickle=False) as data:
    y, v, tags = data['submitted'], data['verified'], data['tags']
    error_rate = np.mean(y != v)
    target = np.flatnonzero(tags != 'O')
    a, b = np.isin(y, target), np.isin(v, target)
    f1 = 2 * np.sum((y == v) & a) / (a.sum() + b.sum())
```

We report **token-level named-entity recognition (NER) micro-F1** over the eight
entity BIO tags. The `O` tag means "outside a named entity" and therefore is not
an entity class. This target measures how accurately entity-bearing tokens are
labeled, without rewarding the many correctly labeled background tokens.
Including all tags in single-label micro-F1 would instead reduce the score to
overall token accuracy, duplicating the information in the mislabel rate.

All `O` tokens remain in the audit population. An entity label assigned to a
verified `O` token counts as a false positive; a submitted `O` label on a verified
entity token counts as a false negative. Thus, the score captures both spurious
and missed entity labels. Matching requires the exact BIO tag: `B-PER` and
`I-PER`, for example, are distinct classes. This is a **token-level** metric,
not the entity-span F1 used for evaluating complete extracted mentions.
Expected F1 is **0.9362349270513437**. Submitted and verified entity-token counts
are 50,711 and 51,210.

| Array | Meaning |
| --- | --- |
| `submitted`, `verified` | Integer tag IDs, shape `(301353,)` |
| `tags` | `O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC`, in this order |
| `record_id` | `split:line`, where line is the 1-based corrected-file line number, including separators |
| `split` | `train`, `dev`, or `test` |
| `sentence` | Zero-based corrected sentence index; resets within each split |

Load the arrays directly without sorting or shuffling to preserve seed-level
record ordering. This is an input release; experiment runners will be added separately.

File SHA-256:
`6fb7bfb5d3b74de9b60db5ff066a8acdb568697d9fcbeed103f0ffb23d6e1bf7`.
The ordered two-column `(submitted, verified)` array serialized as little-endian
64-bit integers in row-major order has SHA-256
`e87661db9f6788471d69a7e709036c1df3d87a88a4dcd86c3b6a176946c76dca`.

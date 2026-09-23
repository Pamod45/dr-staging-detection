# Section 3: Data Splitting and Augmentation

## 3.1 Splitting
Near-duplicate groups: this run found 126 groups (260 images), 4x the original's 30/65. Traced to a
`uint8` overflow bug in the original's Hamming distance calculation; this version's `float32` cast is
mathematically correct, so 126/260 is the accurate count, not a looser detection.

Split: 70/15/15, stratified by grade and source, duplicate groups kept intact.

| Split | No_DR | Mild | Moderate | Severe | PDR | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train (used) | 4,482 | 447 | 3,196 | 170 | 654 | 8,949 |
| Validation | 887 | 91 | 644 | 33 | 130 | 1,785 |
| Test | 897 | 92 | 637 | 33 | 129 | 1,788 |
| IDRiD | 118 | 23 | 164 | 90 | 60 | 455 |

Training imbalance: 26.4:1.

## 3.2 Class weights
Effective-number weights (beta 0.999) computed but not used in this run (`used in training: False`), since
`C.LOSS = 'focal'`, matching the original's final winning choice.

## 3.3 tf.data pipeline
Augmentation, applied live during training: rotate and zoom in one step, flip both axes, contrast, then
brightness, then per-channel colour jitter, background re-masked to black afterward. Validation, test and
IDRiD are unaugmented and deterministic.

`C.CHANNEL = 0.05` is notably smaller than the original notebook's measured colour jitter
([0.20, 0.12, 0.12]), worth checking if this was intentional.

## 3.4 Augmentation preview
Visual check only, no result to report.
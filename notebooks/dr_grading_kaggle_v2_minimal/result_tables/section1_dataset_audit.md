# Section 1: Dataset Audit

## 1.1 Labels, IDRiD correction, class distribution
Loads DDR and IDRiD labels, tags DDR by capture source (hyphen/timestamp), corrects IDRiD labels against
the official source. No new measurement here, carries forward the original notebook's audit.

## 1.2 Sample images per grade
Visual check only, one raw image per grade for DDR and IDRiD. No result to report.

## 1.3 Key findings carried over
- Imbalance: 26.6:1 (No_DR to Severe).
- Two capture sources with different grade mixes (hyphen holds 99.9% of PDR).
- 30 near-duplicate groups, 65 images.
- IDRiD rehost labels disagreed with the official source on 27 images (5.9%); official labels used.
- Severe is 1.9% of DDR but 19.8% of IDRiD, a real prevalence gap.
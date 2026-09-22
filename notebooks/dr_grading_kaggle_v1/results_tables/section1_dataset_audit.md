# Section 1: Dataset Audit

## 1.9 Native vs 512px sharpness

**Result.** Native images score lower on sharpness (median 53.4) than 512px images (113.8), and 12 of the 20 blurriest images are native despite being only 17% of the sample. This is a measurement artefact from shrinking, not real blur, so **no denoising is applied**.

One real difference remains: hyphen images are softer than timestamp images in both size groups. This is a second measured cohort difference, alongside colour, and is the reason Ben Graham normalisation is rejected in 2.5.

## 1.13 IDRiD image properties

**Result.** All 455 IDRiD images are 4288×2848, aspect ratio 1.506, RGB, 228-826 KB - one camera, one resolution, no mixed processing history.

| | R | G | B |
|---|---:|---:|---:|
| DDR hyphen | 134.8 | 84.5 | 41.0 |
| DDR timestamp | 153.4 | 91.9 | 45.7 |
| **IDRiD** | **160.1** | **77.8** | **24.6** |

IDRiD's blue is about 40% lower than either DDR source. `fov_w = 0.8` means IDRiD has black bars on the sides, so a fixed mask would fail here - the retina area must be detected per image.

## 1.14 IDRiD orientation and blur by severity

**Result.** Optic disc sits right in 52% of IDRiD images, left in 48% - close to DDR's 54/46, so horizontal flip stays justified.

Sharpness falls as severity rises: median 58.4, 52.0, 38.7, 43.8, 32.9 for grades 0-4. PDR images are about 44% blurrier than healthy ones, likely real (long-term diabetes can cloud the eye). This only affects the test set and cannot corrupt training.

## 1.16 Near-duplicate threshold

**Result.** Testing several similarity thresholds, the count of duplicate pairs rises steadily from d≤4 to d≤12 without groups merging, then merges start at d≤16. **T = 8** is chosen as the stable middle value.

`ddr_dup_groups.csv` is saved: one row per DDR image, with a group ID shared by near-duplicates. Section 3.1 uses this to keep whole groups in one split.

## 1.16 Duplicate breakdown by grade

**Result.** 30 groups, 65 images, 0.52% of DDR.

| Grade | Duplicated | Total | Rate |
|---|---:|---:|---:|
| 0 No_DR | 46 | 6,266 | 0.73% |
| 1 Mild | 4 | 630 | 0.63% |
| 2 Moderate | 9 | 4,477 | 0.20% |
| 3 Severe | 0 | 236 | 0.00% |
| 4 PDR | 6 | 913 | 0.66% |

Severe has no duplicates, so removing them does not change the class imbalance. 13 of the 30 groups mix both image sources, so filename style is not a perfect proxy for source. 2 groups have conflicting labels, giving a label-error floor of about 0.03%.

## 1.4 Filename suffix vs grade

**Result.** The hyphen filename suffix agrees with the grade on 4,467 of 7,099 images = 62.9%, against a 50.0% majority baseline. Too loose to check labels, but it proves filename leakage is real (a model could reach 62.9% from the filename alone), so **filenames are never used as a model input**.

## 1.20 Section 1: Decisions and limitations

| Decision | Reason |
|---|---|
| Train on DDR, use IDRiD only as an outside test | Both use the same grading scale (ICDR). IDRiD's different colours and disease mix make it a real test of how well the model works on new data |
| Use all 455 IDRiD images as the test set | The model never trains on IDRiD, so every image is new to it (1.17) |
| Use the official IDRiD labels | 27 labels in the Kaggle copy were wrong when checked against the original source (1.12) |
| Find the retina area separately for each image | DDR images are full-frame (`fov_w = 1.0`), but IDRiD images have black bars on the sides (`fov_w = 0.8`) (1.6, 1.13) |
| Apply a circular mask inside the retina edge | 20-35% of each image is black, and the edge has notches and marks that could mislead the model (1.5) |
| Pad to a square, then resize to 512px or smaller | 83% of images are already 512×512. Stretching the other 17% would change the shape of lesions (1.3) |
| Even out brightness and colour | The two DDR sources differ by about 12% in brightness, and IDRiD's blue is 40% lower than both (1.6, 1.13) |
| Give rare grades more weight and add extra varied copies of their images | The biggest class has 26.6 times more images than the smallest, and three classes are very small (1.2) |
| Flip images left to right | Both datasets already have a mix of left and right (54/46 and 52/48) (1.8, 1.14) |
| **No denoising** | The blurry images were caused by resizing, not bad focus (1.9) |
| Split by both grade and image source | Image source can hint at the grade for the rare classes (1.4) |
| Keep duplicate groups together when splitting | We found 30 near-duplicate groups (1.16) |
| Judge with per-class recall, F1, confusion matrix and QWK, not accuracy | The classes are very uneven, the grades are ordered, and the two test sets have different disease mixes |

**Limitations to carry into the report**

1. No patient identifier in DDR, so a patient-grouped split is impossible. Duplicate detection catches retakes but not fellow-eye pairs.
2. DDR labels are unvalidated - no primary source to check against, unlike IDRiD.
3. DDR images have an inconsistent preprocessing history (most downsampled, some not).
4. The "ungradable" category was removed from the DDR copy used, so no quality gate is possible.
5. The cohort (source) confound cannot be removed, only measured and reported.
6. IDRiD healthy cases were over-removed (50 of 61), so prevalence-sensitive metrics are not directly comparable to published IDRiD results.
7. Cross-dataset texture stats are unreliable due to different resampling histories.
8. DDR timestamp grades 1 and 3 hold only 33 and 41 images - too few to characterise well.
# Section 6: Model Evaluation and Performance Analysis

## 6.1 Held-out test performance

**Result. The test score is close to validation - the model transfers well, not overfit to the checkpoint choice.**

| Set | QWK | Severe recall | Mild recall |
|---|---:|---:|---:|
| validation (picked checkpoint) | 0.8682 | 0.556 | 0.330 |
| **internal test (held out)** | **0.8603** | 0.485 | 0.307 |

QWK drops only 0.0079, inside noise (~0.008). AUC is much higher than recall for Mild (0.881 vs 0.307) and Severe (0.961 vs 0.485) - the model ranks these grades well but rarely picks them as the top answer. 92.0% of all mistakes are within one grade.

**Referral (grades 2-4):** sensitivity 0.863, specificity 0.927, 110 of 805 referable cases missed at the default rule.

## 6.2 Normalisation arm at final settings, and the external test

**Result. `none` is clearly better than `clahe_lab`; IDRiD shows a real, expected drop.**

Retrained fairly at the same final settings, `clahe_lab` scores QWK 0.8368 vs `none`'s 0.8682, with Severe/Mild recall collapsing - confirming `none` as the right choice (the earlier 4.5 "tie" was only true at the old settings).

| Set | QWK | Referral sensitivity | Referral specificity | Macro AUC |
|---|---:|---:|---:|---:|
| DDR internal test | **0.8603** | 0.863 | 0.927 | 0.9485 |
| **IDRiD (none)** | 0.7698 | **0.975** | 0.631 | 0.8825 |

Referral sensitivity actually rises on IDRiD, but specificity falls hard - the model over-refers on unfamiliar images. Ranking quality also drops (macro AUC), and the model is *more* confident when wrong on IDRiD than on DDR, so its own confidence is a weaker guide there.

## 6.3 Grad-CAM shortcut check

**Result. The model looks inside the retina, on both DDR and IDRiD.**

Across 13 checked cases, at least 82.6% of model attention sits inside the retina; outside-mask attention never exceeds 12.7%. Correct predictions are confident and focused. Failure cases (missed referrals) are less focused but still look at the retina, not the mask edge or background - so the failures are misjudgement, not a shortcut.

## 6.4 Decision rules on top of the trained model

**Result. A referral threshold helps on DDR but is too aggressive for IDRiD. Per-class weighting doesn't hold up off validation.**

**Referral threshold 0.25** (chosen on validation to keep sensitivity ≥0.95):

| Set | Rule | Sensitivity | Specificity | Missed |
|---|---|---:|---:|---:|
| DDR test | argmax (0.50) | 0.840 | 0.955 | 129 |
| DDR test | threshold 0.25 | 0.944 | 0.796 | 45 |
| IDRiD | argmax (0.50) | 0.968 | 0.681 | 10 |
| IDRiD | threshold 0.25 | 0.997 | 0.546 | 1 |

Works as intended on DDR (missed cases drop, false referrals rise - the right trade for screening). On IDRiD it over-refers (82.9% of images referred), since IDRiD's sensitivity was already high.

**Per-class weighting:** small gains found on validation (0.002-0.005 QWK, inside noise) did not hold on the test set or IDRiD - **rejected**.

## 6.5 Abstention

**Result. Abstaining on the least-confident ~15-24% of cases raises accuracy and clears almost all missed referrals.**

Confidence separates correct from wrong well on DDR (AUC 0.834) but weakly on IDRiD (AUC 0.637). Threshold **0.55** chosen on validation to keep accuracy ≥0.90 among graded cases.

| Set | Coverage | Accuracy (kept) | Missed referrals (auto) |
|---|---:|---:|---:|
| DDR test | 84.7% | 0.904 | **37** (down from 45) |
| IDRiD | 76.3% | 0.706 | **0** (down from 1) |

Abstained cases are disproportionately referable (51.4% vs 44.9% overall), so the model is picking out real hard cases. Mild is the grade it's least sure about (51.1% abstained).

## 6.6 Error analysis and final system

**Result. Training was healthy; the three decision layers together cut missed referrals from 129 to 37.**

Train-validation gap is 0.033 at the end vs 0.104 for the oversampled arm (5.3), which had memorised repeated images. Of 293 test errors, 51.2% are 1 grade off; the model under-grades more than over-grades (175 vs 118), but only 3 of 165 sight-threatening cases fall all the way to No_DR/Mild. Accuracy differs between the two DDR sources (0.783 vs 0.906) - not confirmed as a shortcut (per 6.3), but reported as a limitation.

**The three layers, combined:**

| Layer | Graded | Accuracy | Referral sensitivity | Missed referrals |
|---|---:|---:|---:|---:|
| argmax only | 100.0% | 0.836 | 0.840 | 129 |
| + referral threshold 0.25 | 100.0% | 0.836 | 0.944 | 45 |
| + abstention 0.55 | 84.7% | 0.904 | 0.944 | **37** |

## 6.7 Section 6: Decisions and Conclusion

| Decision | Setting | Reason |
|---|---|---|
| Report held-out test, not validation | QWK 0.8603 | Validation picked the checkpoint, so its own score is optimistic (6.1) |
| Retrain clahe_lab at final settings before comparing | QWK 0.8368 vs 0.8682 | Earlier comparison mixed several changes at once (6.2) |
| **Use `none`, not `clahe_lab`** | - | Clearly worse at final settings (6.2) |
| **Referral threshold: 0.25** | fitted on validation | Missed referrals fall 129→45 on test (6.4) |
| **Per-class weighting rejected** | - | Gains didn't survive off validation (6.4) |
| **Abstention threshold: 0.55** | fitted on validation | Cuts missed referrals to 37, accuracy 0.904 on graded cases (6.5) |

**Final system performance**

| | DDR internal test | IDRiD external |
|---|---|---|
| Grading QWK | 0.860 (0.921 on graded) | 0.770 |
| Referral sensitivity @ 0.25 | 0.944 | 0.997 |
| Silently missed referrals | 37 of 805 | 0 of 313 |

**Limitations to carry into the report**
1. IDRiD needs its own thresholds - the 0.25/0.55 values fitted on DDR over-refer and under-perform on IDRiD
2. Accuracy still differs between the two DDR sources, even though Grad-CAM found no shortcut evidence
3. Mild remains the hardest grade - genuine overlap with neighbouring grades, not a fixable threshold problem
4. All operating points were fitted once on a single validation set, not cross-validated
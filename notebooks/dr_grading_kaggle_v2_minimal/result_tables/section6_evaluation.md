# Section 6: Model Evaluation

## 6.1 DDR internal test
Accuracy 0.869, QWK 0.887, macro F1 0.730. Mild and Severe remain the weakest grades, though both improved
over phase 1. Most Moderate errors go to No_DR, not far misses.

## 6.2 IDRiD external test
Accuracy 0.668, QWK 0.762, macro F1 0.605, a clear drop on unseen external data. Referral sensitivity is
higher on IDRiD than DDR (0.990 vs 0.891), but specificity is far weaker (0.624 vs 0.941), the model
over-flags rather than under-detects on IDRiD.

## 6.3 Grad-CAM
Attention stays inside the retina circle for the cases checked, no evidence of a shortcut on the mask edge
or background.

## 6.4 Prediction grid and single-image demo
Visual check and a reusable single-photo inference function, useful for the coursework's video
demonstration deliverable.

## 6.5 Referral threshold
Chosen on validation: **0.35**. Helps DDR test as intended (sensitivity 0.880 to 0.936, missed referrals
nearly halve, 96 to 51). Does little for IDRiD, whose sensitivity was already high (0.984 at default), and
pushes its specificity down further (0.638 to 0.560, referred % up to 82.2%). A threshold tuned on DDR does
not transfer cleanly to IDRiD's different disease mix.

## 6.6 Abstention
Added to address the IDRiD over-referral problem found in 6.5. Not yet run; no result to report.

## 6.7 Run summary
Saves results, config and probability arrays for the run. Current saved output predates the abstention
addition, so `abstention_threshold` and related fields aren't in it yet.

## 6.8 Comparing runs
Four runs on record. This run (768px, full data) has the best test QWK (0.887) and accuracy (0.869) of the
four. A 512px run with EyePACS data merged reaches slightly higher referral sensitivity (0.954 vs 0.936),
at a real cost to Mild and Severe recall (0.120 and 0.273, both far weaker than this run's).
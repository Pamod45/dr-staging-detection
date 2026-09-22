# Artefacts Inventory

## Root files

- **`ddr_props.csv`** - per-image measurements: size, aspect ratio, brightness, sharpness, source.
- **`ddr_dup_groups.csv`** - per-image duplicate group ID, so duplicates stay in the same split.
- **`ddr_splits.csv`** - per-image label plus which split it's in: train, validation or test.
- **`ddr_splits_manifest.json`** - settings used to build the split: ratios, seed, so it's reproducible.
- **`class_weight.json`** - the 5 per-grade loss weights used to correct class imbalance.

## pilots/

Trained output-layer ("head") weights for each backbone, plus early training measurements (sections 4.3 and 4.4).

- `head_effnetv2b0_none.weights.h5`
- `head_effnetv2b0_clahe_lab.weights.h5`
- `head_resnet50_none.weights.h5`
- **`pilot_history_none.json`** - per-epoch loss/accuracy from the 4.3 head-only backbone comparison.
- **`finetune_history_none.json`** - per-epoch loss/accuracy from the 4.4 fine-tuning sweep (depth and learning rate).

## models/

The two 4.5 trained models (best checkpoint) and everything needed to compare or reuse them.

- `effnetv2b0_none_best.weights.h5`
- `effnetv2b0_clahe_lab_best.weights.h5`
- **`val_probs_effnetv2b0_none.npy`** - the model's 5-grade probabilities for each validation image, `none` arm (used later for thresholds and abstention).
- **`val_probs_effnetv2b0_clahe_lab.npy`** - same, `clahe_lab` arm.
- **`final_history.json`** - per-epoch loss/accuracy/QWK for both 4.5 runs.
- **`final_manifest.json`** - the settings used to train them (backbone, depth, learning rate, size, arm).

## strategy/

The three 5.1 training-schedule comparisons (fixed, plateau, cosine) and their results.

- `fixed_lr.weights.h5`
- `plateau.weights.h5` - **selected**
- `cosine.weights.h5`
- **`val_probs_fixed_lr.npy`, `val_probs_plateau.npy`, `val_probs_cosine.npy`** - each run's 5-grade probabilities on the validation set.
- **`strategy_history.json`** - per-epoch loss/accuracy/QWK for all three runs, including the learning-rate curve.

## resolution/

The three 5.2 input-size comparisons (224, 380, 512 px) and their results.

- `res_224.weights.h5`
- `res_380.weights.h5`
- `res_512.weights.h5` - **selected**
- **`val_probs_224.npy`, `val_probs_380.npy`, `val_probs_512.npy`** - each run's 5-grade probabilities on the validation set.
- **`resolution_history.json`** - per-epoch loss/accuracy/QWK for all three runs.

## balancing/

The two 5.3 class-imbalance comparisons (oversampling, focal loss), at 512 px. Class weights reused 5.2's run rather than retraining.

- `oversampled.weights.h5`
- `focal.weights.h5` - **selected**
- **`val_probs_oversampled.npy`, `val_probs_focal.npy`** - each run's 5-grade probabilities on the validation set.
- **`balancing_history.json`** - per-epoch loss/accuracy/QWK for both runs.

## final/

The two 5.4 full-budget candidates (focal, focal_alpha), plus 6.2.0's fair `clahe_lab` retrain of the winner.

- `focal.weights.h5` - **chosen final model**
- `focal_alpha.weights.h5`
- `focal_clahe_lab.weights.h5` - `clahe_lab` retrained at the exact same settings as `focal`, for the fair 6.2.0 comparison.
- **`val_probs_focal.npy`, `val_probs_focal_alpha.npy`, `val_probs_focal_clahe_lab.npy`** - each run's 5-grade probabilities on the validation set.
- **`final_history.json`** - per-epoch loss/accuracy/QWK for `focal` and `focal_alpha`.
- **`history_focal_clahe_lab.json`** - per-epoch loss/accuracy/QWK for the `clahe_lab` retrain.
- **`final_manifest.json`** - the settings used for the chosen final model (backbone, 512px, focal loss, `none` arm).

## test/

The final model's honest, one-time predictions on the 1,786 held-out DDR test images (section 6.1).

- **`test_probs_focal.npy`** - the model's 5-grade probabilities for each test image, first ever look at this set.
- **`test_labels.npy`** - the true grades for the same 1,786 images, so predictions can be scored against them.

## external/

The model's predictions on all 455 IDRiD images, the real generalisation test (section 6.2).

- **`ext_probs_focal_none.npy`** - 5-grade probabilities for each IDRiD image, `none` arm model.
- **`ext_probs_focal_clahe_lab.npy`** - same, `clahe_lab` arm model.
- **`ext_labels.npy`** - the true IDRiD grades, to score predictions against.

## gradcam/

Contains 6.3 Grad-CAM outputs for DDR success/failure cases, along with IDRiD dataset images.

## thresholds/

The chosen referral decision rule (section 6.4).

- **`operating_points.json`** - the referral threshold (0.25), the sensitivity floor it was chosen to meet, and the two rejected per-class weight sets, for reference.

## abstention/

The chosen "not sure, ask a human" rule (section 6.5).

- **`abstention.json`** - the abstention threshold (0.55), which confidence measure it uses, and the accuracy target it was fitted to.
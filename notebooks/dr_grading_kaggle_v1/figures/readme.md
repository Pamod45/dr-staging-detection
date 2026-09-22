# Figures

| File | Section | What it shows |
|---|---|---|
| `sec2.3_fov_crop_mask_before_after.png` | 2.3 | 4 sample images, before (top row) and after (bottom row) FOV crop, circular mask and resize to 512px |
| `sec2.4_normalisation_candidates_per_grade.png` | 2.4 | One image per ICDR grade, shown under each of the 4 normalisation candidates (`none`, `clahe_green`, `clahe_lab`, `bengraham`) |
| `sec3.3_rotation_interpolation_comparison.png` | 3.3 | One image rendered 5 ways (resize-only, rot90, one-warp LINEAR, one-warp CUBIC, two-step), for comparing detail loss |
| `sec3.4_photometric_jitter_examples.png` | 3.4 | Source image plus 5 jittered copies, with the gain and contrast values used for each |
| `sec3.5_augmentation_draws.png` | 3.5 | One source image, alongside 8 augmented draws (`clahe_lab` arm), showing the pipeline's eval path vs random rotation/zoom/flip/jitter |
| `sec4.3_pilot_loss_accuracy_curves.png` | 4.3 | Training and validation loss/accuracy curves for the head-only pilot (EfficientNetV2-B0 vs ResNet50) |
| `sec4.4_finetune_val_loss_accuracy.png` | 4.4 | Validation loss and accuracy curves across the fine-tuning sweep (backbone depth, learning rate) |
| `sec5.2_resolution_qwk_f1_speed.png` | 5.2 | Validation QWK, macro F1 and seconds/epoch plotted against input resolution (224/380/512px) |
| `sec5.4_final_candidates_curves.png` | 5.4 | Training curves for the two final candidates (focal, focal_alpha) |
| `sec6.1_confusion_matrix_confidence.png` | 6.1 | Row-normalised confusion matrix (internal test) alongside the max-softmax confidence distribution |
| `sec6.2_idrid_confusion_recall.png` | 6.2 | IDRiD confusion matrices and per-class recall, compared across arms |
| `sec6.4_roc_precision_recall_curves.png` | 6.4 | ROC curve and precision-recall curve for the referable (grades 2-4) decision, with the chosen threshold marked |
| `sec6.6a_final_training_curves.png` | 6.6 | Final model's accuracy, focal loss and selection-metric (QWK/F1) curves across training |
| `sec6.6b_oversampled_arm_loss.png` | 6.6 | Loss curve for the 5.3 oversampled arm, used as the overfitting contrast case |
| `sec6.6c_oversampled_arm_metric.png` | 6.6 | Error distance histogram, confidence-by-outcome histogram (with abstention line), and recall by grade |
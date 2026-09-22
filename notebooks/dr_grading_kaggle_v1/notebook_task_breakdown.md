# Notebook Section & Subsection Tasks

## Section 1: Dataset Audit
**Main purpose:** check both datasets are clean, understand their differences, and find risks before training.

**1.1 DDR - label integrity**
1. Load the DDR label CSV and check its columns, row count and data types.
2. Check for duplicate or missing label rows.
3. Cross-check every labelled image actually exists on disk, and flag any unlabelled files.

**1.2 DDR - class distribution**
1. Confirm the label set matches the expected 5 grades.
2. Count images per grade and their percentage share.
3. Compute the imbalance ratio (largest class to smallest).
4. Plot the class counts as a bar chart.

**1.3 DDR - image properties**
1. Read every image's width, height, colour mode and file size.
2. Tag each image by filename convention (hyphen or timestamp source).
3. Compute aspect ratio and megapixels per image.
4. Save the full property table to `ddr_props.csv`.
5. Summarise resolution, aspect ratio, size and file-size spread.

**1.4 DDR - groups within the dataset (cohorts)**
1. Compare class distribution between the two filename sources.
2. Check how much of each grade comes from each source.
3. Compare median resolution and file size between sources.

**1.5 DDR - visual inspection**
1. Pick sample images across sources, grades and resolutions.
2. Display them side by side for a manual sanity check.

**1.6 DDR - colour and illumination**
1. Define a function to measure per-image brightness (R/G/B), colour spread and field-of-view size.

**1.7 DDR - separating cohort effect from grade effect**
1. Compute mean colour stats grouped by both source and grade.
2. Check sample sizes behind each group, to judge how reliable the comparison is.

**1.8 DDR - focus quality and eye side**
1. Measure sharpness (blur score) and which side the optic disc sits on, per image.
2. Summarise blur score distribution overall and by source.
3. List the 10 blurriest images.
4. Report the left/right disc-side split.

**1.9 DDR - blur versus resolution**
1. Split images into 512px vs native-resolution groups.
2. Compare blur score between the two groups.
3. Check how the two groups split across sources.
4. Check which group the 20 blurriest images belong to.

**1.10 IDRiD - label integrity**
1. Load the IDRiD label CSV, check columns and row count.
2. Check for duplicate or missing labels.
3. Cross-check every labelled image exists on disk.

**1.11 IDRiD - columns, splits and distribution**
1. Clean up the CSV columns and rename them.
2. Derive the train/test split from the filename.
3. Check for missing image-ID numbers within each split.
4. Show the grade distribution before correction.

**1.12 IDRiD - validate and correct labels against the primary source**
1. Merge the working labels against the official source labels.
2. Count and list every mismatch.
3. Check which direction the errors run, and how many diseased cases were mislabelled healthy.
4. Overwrite the working labels with the corrected ones.

**1.13 IDRiD - image properties and dataset difference vs DDR**
1. Read every IDRiD image's size, mode and file size.
2. Compute aspect ratio.
3. Measure the same colour/FOV stats used for DDR.
4. Compare IDRiD's stats directly against both DDR sources in one table.

**1.14 IDRiD - focus quality and eye side**
1. Measure blur score and disc side for every IDRiD image.
2. Report the disc-side split.
3. Compare blur score across IDRiD's own grades.

**1.15 Cross-dataset class distribution**
1. Compare class share (%) between DDR and IDRiD side by side.
2. Compute the ratio between the two, per grade.

**1.16 Near-duplicate detection (DDR)**
1. Compute a perceptual hash for every DDR image.
2. Compute pairwise distance between all hashes.
3. Sweep several distance thresholds and check pair count, image count and largest group size at each.
4. Pick the threshold where groups stay small but real duplicates are still caught.

**1.17 Cross-dataset overlap**
1. Hash every IDRiD image the same way.
2. Compute the minimum distance between any DDR and IDRiD image.
3. Check how many cross-dataset pairs fall under several thresholds.

**1.18 Filename leakage**
1. Extract the numeric suffix from hyphen filenames.
2. Cross-tabulate suffix against grade.
3. Measure how often the suffix alone predicts the grade.
4. Compare that against the majority-class baseline.

**1.19 Export duplicate groups**
1. Build a table of every image with its duplicate group ID and group size.
2. Save it to `ddr_dup_groups.csv`.
3. Assert the table covers every DDR image.

## Section 2: Preprocessing
**Main purpose:** build the image processing pipeline and justify each choice with measurements.

**2.1 Per-image field-of-view detection**
1. Define a function that finds the retina's bounding circle from image content.
2. Measure spill (mask overshoot) and letterbox (retina fill) for each detection.
3. Run detection on a stratified sample across both datasets.

**2.2 Choosing the mask fraction**
1. Define a function measuring how much retina a given mask fraction keeps.
2. Test several mask fractions (0.90 to 1.00) on both datasets.
3. Compare mean and worst-case (5th percentile) retention per fraction.

**2.2.1 Where the retina actually ends**
1. Build a radial brightness profile per image.
2. Find where brightness drops to half the plateau level, as a fraction of the detected radius.
3. Summarise how far the real retina edge sits relative to the detected circle.

**2.2.2 Where the notch reaches**
1. Unwrap the boundary ring of the worst-spill images into a flat strip.
2. Mark the 0.95 and 0.98 mask lines on the strip.
3. Visually check where the camera notch sits relative to those lines.

**2.3 Mask, square crop and resize as one resample**
1. Define the final preprocessing function: mask, crop to square, resize in a single resample.
2. Fix the output size and mask fraction settings.
3. Run it on a few representative images (square DDR, native DDR, timestamp DDR) as a check.

**2.3.1 Confirming the notch is gone from the output**
1. Unwrap the boundary of the processed output images the same way as 2.2.2.
2. Check visually that no bright notch crosses the mask edge line anymore.

**2.4 Normalisation candidates**
1. Define 4 normalisation functions: none, CLAHE on green, CLAHE on LAB lightness, Ben Graham.
2. Apply each to sample images for visual comparison.

**2.5 Ablation - which variants go forward**
1. Measure brightness/colour/texture gap between the two DDR sources under each normalisation variant.
2. Compare how well each variant closes that gap.
3. Decide which variants to carry forward for training.

**2.5.1**
1. Re-test the earlier texture-severity signal at the per-image level, not just as a group average.
2. Check correlation between texture measure and grade, per image.
3. Explain why the ablation's original justification does or doesn't hold.

**2.6 Batch run and Local cache**
1. Set up output folders and a manifest describing the current preprocessing settings.
2. Check if a matching cache already exists; raise an error if settings changed.
3. Define a per-image worker function that preprocesses and saves one image.
4. Run preprocessing across all DDR and IDRiD images in parallel.
5. Track and report success/skip/fail counts.

**2.7 Verify the processed set**
1. Count processed files against expected counts for both datasets.
2. Check the saved manifest still matches current settings.
3. Check for missing or suspiciously small output files.
4. Sample and reload some outputs to check shape and dark-pixel fraction.

## Section 3: Augmentation
**Main purpose:** build the training data pipeline and size each augmentation using measurements, not guesses.

**3.1 Splitting: group-aware and doubly stratified**
1. Merge duplicate-group IDs into the image property table.
2. Define a stratification key combining grade and source.
3. Use grouped stratified splitting so duplicate groups never cross a split boundary.
4. Cache the split to disk with a settings manifest, and reload it if unchanged.

**3.2 Class imbalance: weighting compared against resampling**
1. Count training images per grade.
2. Compute several candidate weighting schemes (inverse, sqrt-inverse, effective-number at different betas).
3. Compare each scheme's resulting loss share per class.
4. Compare weight spread (max:min) across schemes.
5. Report how many times each class would repeat under full oversampling.

**3.3 Geometric augmentation, with its cost measured**
1. Define a combined rotate/scale/flip/resize function in one resample step.
2. Define a fine-detail measure (Laplacian variance) restricted to the retina disc.
3. Define a dark-fraction measure to check for mask leakage.
4. Sample training images and test detail retention across rotation angles and interpolation methods.

**3.4 Photometric jitter sized from the measured confounds**
1. Define a function measuring per-image mean colour inside the mask.
2. Define an AUC-based separability measure between two groups.
3. Build colour-ratio statistics (R/G, B/G, luminance) from raw channel means.
4. Sample images from both DDR sources and from IDRiD.
5. Measure how separable the sources are before and after simulated jitter, to size the amplitude.

**3.5 The `tf.data` pipeline**
1. Define the training augmentation function (geometry, then jitter, then normalisation).
2. Define the fixed evaluation-path function (no randomness).
3. Build a `tf.data` dataset loader supporting both weighted and per-class-balanced sampling.
4. Build train/val/test/external datasets from the saved split.
5. Check batch shape, dtype and value range.
6. Assert the evaluation and external paths are deterministic, and the training path is not.
7. Compare class share seen under the weighted vs balanced sampling arms.
8. Visually show a batch of augmented draws from one source image.

**3.6 Throughput**
1. Time the pipeline under each normalisation arm, for both the evaluation and augmented training paths.
2. Break down per-stage cost (decode, geometry, normalisation, jitter).
3. Compare pipeline throughput against the GPU's own processing speed.

## Section 4: CNN Architecture and Transfer Learning
**Main purpose:** pick a backbone, fine-tuning depth and learning rate, then train the first real model.

**4.1 Runtime, precision, and the model factory**
1. Detect the GPU and enable mixed precision if supported.
2. Define the two candidate backbones with their preprocessing and native input size.
3. Define a reusable model-building function: backbone plus pooling, dropout and a 5-way output.
4. Print each candidate's parameter counts and trainable head size.
5. Run a random-input probe to confirm output shape and valid probabilities.

**4.2 Sanity checks before any real training run**
1. Check that loaded labels align with their images.
2. Check the model's output actually changes with input pixel scale.
3. Check BatchNorm statistics stay frozen while the backbone is frozen.
4. Run a short overfit test on a tiny sample to confirm the model can learn at all.
5. Measure the model's own raw training speed (images per second) for comparison against the pipeline.

**4.3 Backbone pilot - head-only**
1. Train only the output layer for both backbones, backbone frozen.
2. Score each on the validation set: accuracy, macro F1, QWK, per-class recall.
3. Include a majority-class baseline for comparison.
4. Show confusion matrices and training curves for both.

**4.4 Fine Tuning**
1. Warm up the head for each backbone once, so every fine-tuning run starts from the same point.
2. Compare both backbones fine-tuned at one unfreeze depth and learning rate.
3. Sweep unfreeze depth (25% / 50% / 100%) on the winning backbone.
4. Test a lower learning rate on the best depth.
5. Summarise all runs in one table and show the best one's confusion matrix and curves.

**4.5 Final runs - both normalisation arms, checkpointed**
1. Train the chosen backbone/depth/learning-rate setup on the `none` arm.
2. Train the same setup on the `clahe_lab` arm.
3. Save the best-epoch checkpoint by validation QWK for each arm.
4. Save validation probabilities, training history and a settings manifest.
5. Compare the two arms' results side by side.

## Section 5: Training Strategy and Experimental Design
**Main purpose:** test schedule, resolution and imbalance choices properly, then lock in the final model.

**5.0**
1. Reload the saved 4.5 model settings and helper functions.
2. Check all expected saved files exist before running long experiments.

**5.1 Learning-rate schedule and early stopping**
1. Train the same setup under 3 schedules: fixed, reduce-on-plateau, cosine decay.
2. Add early stopping based on validation QWK.
3. Compare final scores, epochs run and training time across the 3 schedules.
4. Plot QWK, loss and learning-rate curves for each.

**5.2 Input resolution ablation**
1. Reuse the chosen schedule at 3 input sizes: 224, 380, 512px.
2. Load the same saved head weights at each size to start fairly.
3. Compare QWK, macro F1, per-class recall and cost per epoch across sizes.
4. Convert resolution to real-world micrometres per pixel, for context against lesion size.

**5.3 Balancing strategy: weights, resampling, or focal loss**
1. Reuse the winning resolution from 5.2.
2. Train with oversampling instead of class weights.
3. Train with focal loss instead of class weights.
4. Compare all three balancing strategies on the same metrics.

**5.4 Final candidates - full budget, and the combination 5.3 left untested**
1. Train the winning balancing strategy (focal loss) at a longer epoch budget.
2. Train a second candidate: focal loss combined with class weights.
3. Compare both, including referral-specific sensitivity, specificity and missed-referral count.
4. Save the final chosen model's weights, probabilities and settings manifest.

## Section 6: Model Evaluation and Performance Analysis
**Main purpose:** get the honest final score, check for shortcuts, and turn the model into a usable decision system.

**6.1 Internal test evaluation**
1. Load the final model and run it once on the held-out DDR test set.
2. Compare test score against the validation score that picked the checkpoint.
3. Report per-class precision, recall, F1 and confusion matrix.
4. Report per-class ROC AUC and macro AUC.
5. Report referral-collapsed sensitivity, specificity and PPV.
6. Report how far off (in grades) the wrong predictions typically are.

**6.2.0**
1. Retrain the `clahe_lab` arm at the exact final settings (resolution, loss, schedule), so it's a fair comparison to the chosen model.

**6.2 External test: IDRiD**
1. Run the final model on all IDRiD images.
2. Run the retrained `clahe_lab` arm on IDRiD too, if available.
3. Compare DDR internal test against IDRiD external test, for both arms.
4. Report IDRiD's per-class results and confusion matrix.
5. Compare per-class recall change between internal and external test.
6. Compare mean confidence, and confidence specifically on wrong predictions, between the two test sets.

**6.3 Grad-CAM: what the model is looking at**
1. Build a Grad-CAM heatmap function for the trained model.
2. Verify the heatmap's predicted class matches the model's real prediction.
3. Measure how much heatmap weight falls inside the retina, on the rim, and outside the mask.
4. Show correct predictions, one per grade, with their heatmaps.
5. Show clinically important failure cases with their heatmaps.
6. Show IDRiD examples, including a wrong prediction, with their heatmaps.

**6.4 Operating points: turning probabilities into decisions**
1. Sweep referral-probability thresholds on validation, measuring sensitivity and specificity at each.
2. Pick the threshold that keeps sensitivity above a target floor while maximising specificity.
3. Apply that fixed threshold to DDR test and IDRiD.
4. Search for per-class probability weights that improve macro F1 or QWK on validation.
5. Test boosting one class (Mild) in isolation, to show the recall/precision trade-off directly.
6. Apply the found weights, unchanged, to DDR test and IDRiD, to check they generalise.

**6.5 Confidence-based abstention**
1. Compare 3 confidence measures (top probability, margin, entropy) for separating correct from wrong predictions.
2. Measure calibration (expected calibration error) on validation, test and IDRiD.
3. Build a coverage curve: accuracy versus how many low-confidence cases are abstained.
4. Pick an abstention threshold on validation that keeps accuracy above a target floor.
5. Apply that threshold, unchanged, to DDR test and IDRiD.
6. Break down abstention rate by grade.
7. Check how abstention changes the missed-referral count.

**6.6 Error analysis and training behaviour**
1. Plot the final model's training/validation accuracy, loss and selection-metric curves.
2. Compare its train-validation gap against the oversampled arm from 5.3.
3. Measure how far off (in grades) the test-set errors are, and whether the model over- or under-grades.
4. Check how many sight-threatening cases were under-graded, and how severely.
5. Check whether accuracy differs by DDR source (cohort), to re-check the 1.4 shortcut risk.
6. Combine the referral threshold and abstention rule into one final decision-layer comparison table.
7. Plot error distance, confidence-by-outcome and per-grade recall together.
# Notebook Section & Subsection Tasks

## Section 0: Setup
**Main purpose:** load libraries, define configuration, and locate the input files.

**0.1 Imports, GPU and mixed precision**
1. Import all libraries used throughout the notebook.
2. Install `imagehash` if not already available.
3. Import TensorFlow and Keras.

**0.2 Config**
1. Define a `Config` class holding every tunable setting: data split, cache size, model input size, batch size, normalisation, model head type, dropout, unfreeze share, loss function, training schedule, and run tag.

**0.3 Paths and input files**
1. Locate the DDR and IDRiD CSVs and image folders under `/kaggle/input`.
2. Assert both image folders exist.
3. Build a cache specification from the config, used later to check whether a matching preprocessed cache already exists.

## Section 1: Dataset Audit
**Main purpose:** check both datasets are clean, understand their differences, and find risks before training.

**1.1 Load labels, correct IDRiD, class distribution**
1. Load DDR labels, tag each image by filename source (hyphen or timestamp).
2. Check every labelled DDR image exists on disk.
3. Load IDRiD labels, clean up columns, derive the train/test split from the filename.
4. Correct IDRiD labels against the official source, replacing wrong grades.
5. Show DDR and IDRiD class counts and percentages side by side.

**1.2 Sample images per grade (raw, before preprocessing)**
1. Pick one random image per grade from DDR.
2. Pick one random image per grade from IDRiD.
3. Display all 10 as a 2x5 grid, DDR on top, IDRiD on bottom.

**1.3 Key findings carried over from the full dataset analysis** *(markdown, not code)*
1. States the imbalance ratio (26.6:1), the two capture sources and their different grade mixes, the near-duplicate count (30 groups, 65 images), the IDRiD label correction count (27 of however many), the Severe prevalence gap between DDR and IDRiD, and a note on ethics and data limits, all carried over from the original, fuller notebook.

## Preprocessing (Section 2)
**Main purpose:** build the image processing pipeline, apply it to every image, and check it works correctly.

**2.1 Preprocessing functions**
1. Define `detect_fov`: find the retina's bounding circle in an image.
2. Define `circle_mask`: build a circular mask at a given fraction of the frame.
3. Define `preprocess`: the full pipeline, detect, trim the radius, centre on a black square canvas, mask, resize to the cache size.
4. Define `apply_clahe`: local contrast boost on lightness only, background re-blacked afterward.

**2.2 Before / after: raw to cropped/masked to CLAHE**
1. Pick 4 example images: DDR grade 0, DDR grade 4, IDRiD grade 0, IDRiD grade 3.
2. Show each one raw, after `preprocess()`, and after CLAHE.
3. Plot a green-channel histogram before and after CLAHE, to confirm contrast actually widened.

**2.3 Build the processed image cache**
1. Define a per-image worker that preprocesses one image and writes it to disk, skipping images already cached.
2. Check whether a complete, matching cache already exists via a manifest.
3. Run preprocessing across all DDR and IDRiD images in parallel.
4. Track and report success/skip/fail counts, assert none failed.
5. Save a manifest recording the cache settings used.

**2.4 EyePACS Mild + Severe as extra training data**
1. Check whether `MERGE_EXTRA` is turned on; skip entirely if not.
2. Load EyePACS labels, filter to only Mild and Severe images.
3. Preprocess those extra images through the same pipeline, cached separately.
4. Drop any that failed or weren't found, kept out of validation, test and IDRiD.

## Section 3: Data Splitting and Augmentation
**Main purpose:** split DDR into train, validation and test without leaking duplicates across splits, then define how training images are randomly varied on the fly during training.

**3.1 Duplicate groups, group-aware split, TRAIN_SIZE subset, optional EyePACS merge**
1. Load or build near-duplicate groups, using a 256-bit perceptual hash if no existing file is found.
2. Split DDR into 70% train, 15% validation, 15% test, stratified by grade and source, duplicate groups kept intact.
3. Reuse an existing split file if one is attached and matches current settings, otherwise build a new one.
4. Assert the split is leak-free.
5. Apply `TRAIN_SIZE`, if configured, to optionally shrink the training pool while protecting rare classes.
6. Optionally merge in EyePACS extra images, training set only.
7. Report final class counts, imbalance ratio and image sources across all splits.

**3.2 Class weights**
1. Compute effective-number class weights (beta 0.999), for optional use if the loss function is set to weighted cross-entropy. Only active when `C.LOSS == 'ce_weighted'`.

**3.3 tf.data pipeline**
1. Define `augment()`: rotate, zoom, flip, then contrast, brightness and per-channel colour jitter, re-masked to black background.
2. Define `load_image()`: read from disk, resize if needed, augment if training, apply CLAHE if configured.
3. Define `make_ds()`: build a `tf.data` dataset from a table.
4. Build a small unaugmented `clean_ds` subset of training images.
5. Build all 5 datasets: train, val, test, IDRiD, clean.
6. Check batch shape, dtype and pixel range.
7. Assert the validation path is deterministic.

**3.4 Augmentation preview**
1. Load one probe training image.
2. Generate 7 randomly augmented versions using the real `augment()` function.
3. Display as a grid, titled with the actual configured amplitudes.

## Section 4: CNN Architecture and Transfer Learning
**Main purpose:** build the model, choosing and justifying the backbone and head.

**4.1 Build the model**
1. Define `build_model()`: EfficientNetV2-B0 backbone (with rescaling built in), pooling, a configurable head (`dense` or `simple`), 5-way softmax output.
2. Keep the backbone frozen and running in inference mode, so its ImageNet BatchNorm statistics stay untouched.
3. Build the model, print its summary and trainable parameter count.
4. Run a random-input probe to confirm output shape and valid probabilities.

## Section 5: Training Strategy
**Main purpose:** define the loss, monitoring and schedule, then run both training phases and check the result isn't overfitting.

**5.1 Loss function and callbacks**
1. Define `sparse_focal()`: focal loss, cross-entropy weighted by `(1 - p_true)^gamma`, so hard examples count more.
2. Choose the loss function based on config (`focal` or plain cross-entropy).
3. Define `CleanTrainMetrics`: accuracy, loss, QWK and F1 on the unaugmented clean training subset, every epoch.
4. Define `ValQWK`: QWK, F1, Mild recall/precision, Severe recall on validation, every epoch.
5. Define `callbacks_for()`: checkpoint by validation loss and separately by validation QWK, early stopping on validation loss, reduce learning rate on plateau, CSV logging.

**5.2 Phase 1: frozen backbone, train the head**
1. Compile the model with the chosen loss and the phase 1 learning rate.
2. Train for the configured number of epochs, backbone frozen.
3. Log accuracy, loss, clean metrics and validation QWK each epoch.

**5.3 Phase 2: unfreeze the top of the backbone and fine-tune**
1. Unfreeze the top configured share of backbone layers.
2. Recompile at the phase 2 learning rate.
3. Train with early stopping, checkpointing and learning-rate reduction active, up to the configured maximum epochs.
4. Restore the best checkpoint by validation loss.

**5.4 Accuracy and loss curves**
1. Plot accuracy and loss for augmented training, clean training and validation together, across both phases.
2. Mark the phase 1/phase 2 boundary and the best epoch.

**5.5 Gap between training and validation**
1. Compute the accuracy gap between augmented training and validation, and between clean training and validation, at every epoch.
2. Report both gaps specifically at the best epoch, separating real overfitting from the gap caused by augmentation and dropout alone.

## Section 6: Model Evaluation
**Main purpose:** get honest performance on data the model never trained or tuned on, check what it's actually looking at, and turn its probabilities into a usable referral decision.

**6.1 DDR test set: metrics, classification report, confusion matrix**
1. Define `metrics()`: accuracy, QWK, macro F1, per-class recall, referral sensitivity/specificity/AUC, all in one call.
2. Define `plot_cm()`: confusion matrix plotted as both raw counts and row-normalised percentages.
3. Predict on validation and test, store both in `R`.
4. Print headline test numbers, the full classification report, and the confusion matrix.

**6.2 IDRiD external test**
1. Predict on all 455 IDRiD images.
2. Store IDRiD metrics in `R`.
3. Print the classification report and confusion matrix, and plot it.
4. Print a combined comparison table across validation, test and IDRiD.

**6.3 Grad-CAM**
1. Build a Grad-CAM function tracing gradients back through the backbone's final feature map.
2. Pick the most confident correct prediction per grade, and the most confident wrong predictions.
3. Overlay the heatmap on each image, for both groups.

**6.4 Prediction grid and single-image inference**
1. Show 12 random test images with true grade, predicted grade and confidence, coloured by correctness.
2. Define `predict_single_image()`: run the full pipeline (or skip it if already cached) on one arbitrary photo, predict, and show the image with a probability bar chart.
3. Demonstrate it once on an IDRiD image.

**6.5 Referral threshold**
1. Define `referral()`: sensitivity, specificity, missed count and referral rate at a given probability threshold.
2. Sweep thresholds on validation, pick the highest one still keeping sensitivity at or above 95%.
3. Compare the default threshold (0.5) against the chosen one, across validation, test and IDRiD.
4. Store the chosen threshold's sensitivity and specificity into `R`.

**6.6 Confidence-based abstention**
1. Define `coverage_curve()`: accuracy among the kept cases, at increasing coverage levels, sorted by confidence.
2. Pick the lowest confidence cutoff on validation that still keeps accuracy at or above 90%.
3. Apply that threshold to validation, test and IDRiD, reporting coverage, accuracy kept, and missed referrals among the auto-graded cases.

**6.7 Save the run and print a summary**
1. Build a flat results dictionary: run identity, split, training length, best epoch, the accuracy gap, referral threshold, and every metric from `R`.
2. Save `results.json` and `config.json`.
3. Save the raw prediction probability arrays for test and IDRiD.
4. Print a readable run summary.

**6.8 Compare all runs**
1. Read every saved `results.json` found under the working directory and attached inputs.
2. Build a comparison table across key metrics, sorted by test QWK.
3. Warn if runs being compared used different data splits.
4. Plot validation accuracy and the clean-train/validation gap, per run.
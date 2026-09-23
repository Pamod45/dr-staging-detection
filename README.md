## Notebooks

- **`dr_grading_kaggle_v1/`** — the original, fully detailed notebook. Covers dataset audit, preprocessing
  justification, augmentation sizing, architecture selection and training-strategy comparisons (schedule,
  resolution, class balancing) as measured experiments, each backed by evidence before a decision is made.
  Final model: EfficientNetV2-B0, 512px, focal loss. DDR test QWK 0.860, IDRiD external QWK 0.770.

- **`dr_grading_kaggle_v2_minimal/`** — a simplified rebuild of the same pipeline, keeping the justified
  decisions from v1 (retina detection, masking, augmentation choices, EfficientNetV2-B0) but skipping the
  comparison work, adopting those choices directly instead of re-deriving them. Adds a few things v1 didn't
  have: a larger 768px cache, an optional EyePACS merge for the rarest classes, and confidence-based
  abstention. Includes 4 training runs at different configurations. Best run (`04_full_dense_none_focal_768`):
  DDR test QWK 0.887, IDRiD external QWK 0.762.
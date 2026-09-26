# Diabetic Retinopathy Grading

Grades retinal (fundus) photographs on the five-step ICDR scale with EfficientNetV2-B0, and
serves the model in a Streamlit app that explains its result.

> For research and education. Not a diagnostic device; results must not be used for clinical
> decisions.

## Notebooks

- **`dr_grading_kaggle_v1/`** - the full, detailed notebook: dataset audit, preprocessing,
  augmentation, architecture selection and training-strategy experiments (schedule, resolution,
  class balancing), each measured before a decision is made. Final model: EfficientNetV2-B0,
  512 px, focal loss. DDR test QWK 0.860, IDRiD external QWK 0.770.
- **`dr_grading_kaggle_v2_minimal/`** - a leaner rebuild that adopts v1's decisions directly and
  adds a 768 px cache and an optional EyePACS merge for rare grades. Four training runs. Best run
  (`04_full_dense_none_focal_768`), used by the app: DDR test QWK 0.887, IDRiD external QWK 0.762.

## The app

| Page | What it shows |
|---|---|
| About diabetic retinopathy | The five grades, expert-annotated lesions per grade, referral, datasets |
| Screening | Grade, confidence, referral, Grad-CAM, explanation, PDF report, chat |
| How it was built | Preprocessing, augmentation, architecture and training, on a real image |
| Results and decisions | Why each design choice was made, final results, referral threshold |
| Model comparison | Models that differ in one setting, side by side |

## Run it

Requires Python 3.11 or 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Model files** are not in the repository (about 61 MB each). Put them in `models/` using the
layout in `APP_PLAN.md`, then check:

```bash
python scripts/check_models.py
```

**Chat (optional).** Create `.streamlit/secrets.toml` with `GEMINI_API_KEY = "your-key"`. This
file is gitignored. Without a key the app works and the chat is hidden.

```bash
streamlit run app/main.py
python -m pytest -q
```

## Scripts

- `scripts/check_models.py` - are all model and figure files in place?
- `scripts/check_screening.py --ddr <folder> --per-grade 4` - does the app reproduce the notebook
  on real test images?
- `scripts/pick_lesion_examples.py --idrid <folder>` - finds the official grade of each IDRiD
  segmentation image and copies one per grade for the Home page.

## Data and credits

- **DDR** - Li T et al., *Information Sciences* 2019. Training, validation and internal test.
- **IDRiD** - Porwal P et al., *Data* 2018, CC BY 4.0. External test set, and the
  expert-annotated lesion images on the Home page.
"""Copy one IDRiD grading photograph per grade into content/samples/grades/grade_N.jpg, and a
Moderate one to content/samples/sample.jpg. IDRiD is CC BY 4.0, so these can be shown on the
public app. Grades come from the official IDRiD training labels.

    python scripts/pick_grade_photos.py --idrid "C:/.../Datasets/IDiRD"
    python scripts/pick_grade_photos.py --idrid "C:/.../Datasets/IDiRD" --pick 2=3

--pick G=K takes the K-th image (from 0) of grade G instead of the first, if one looks poor.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"^IDRiD_\d{3}\.jpg$", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--idrid", required=True, type=Path)
ap.add_argument("--pick", action="append", default=[], help="G=K, e.g. 2=3")
args = ap.parse_args()

csvs = [p for p in args.idrid.rglob("*.csv")
        if "training" in p.name.lower() and "label" in p.name.lower()]
if not csvs:
    sys.exit(f"No training labels CSV found under {args.idrid}. CSVs seen: "
             f"{[p.name for p in args.idrid.rglob('*.csv')]}")
labels = pd.read_csv(csvs[0])
labels.columns = [c.strip() for c in labels.columns]
print(f"labels: {csvs[0]}")

# official training images: IDRiD_001.jpg style names in a folder whose path says Training,
# outside the segmentation part
images = {p.stem: p for p in args.idrid.rglob("*.jpg")
          if NAME.match(p.name) and "training" in str(p.parent).lower()
          and "segmentation" not in str(p).lower()}
if not images:
    sys.exit(f"No IDRiD_###.jpg training images found under {args.idrid}")
print(f"images: {len(images)} training photographs")

picks = {int(g): int(k) for g, k in (s.split("=") for s in args.pick)}
out = ROOT / "content" / "samples"
(out / "grades").mkdir(parents=True, exist_ok=True)

def nth(grade: int, k: int) -> Path:
    names = [n.strip() for n in labels.loc[labels["Retinopathy grade"] == grade, "Image name"]]
    names = [n for n in names if n in images]
    if k >= len(names):
        sys.exit(f"grade {grade} has only {len(names)} images; --pick {grade}={k} is too high")
    return images[names[k]]

for g in range(5):
    src = nth(g, picks.get(g, 0))
    shutil.copy2(src, out / "grades" / f"grade_{g}.jpg")
    print(f"grade {g}: {src.name}")
sample = nth(2, picks.get(2, 0) + 1)
shutil.copy2(sample, out / "sample.jpg")
print(f"sample.jpg: {sample.name} (Moderate)")
"""Optional: does the app reproduce the notebook on real images?

    python scripts/check_screening.py --ddr "<folder with raw DDR jpgs>" --n 10
    python scripts/check_screening.py --ddr "<folder>" --model v1_res_224

Runs the app's input path and model on the first N images of the split each model was scored
on (v1: validation, v2: test) and compares with the notebook's stored probabilities. Small
differences are expected: the notebooks read JPEG-cached images on a GPU in float16, the app
reads raw images on CPU in float32. What must match is the grade.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2           # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C              # noqa: E402
from src import data                     # noqa: E402
from src.runner import ModelRunner       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ddr", required=True, help="folder holding the raw DDR images")
ap.add_argument("--model", default=C.SCREENING_MODEL_ID, choices=sorted(C.MODELS))
ap.add_argument("--n", type=int, default=10)
args = ap.parse_args()

spec = C.MODELS[args.model]
split = pd.read_csv(C.SPLIT_FILES[spec.notebook])
rows = split[split["split"] == spec.eval_set].reset_index(drop=True)
stored = data.probs(args.model)
assert len(rows) == len(stored), "split rows and stored probabilities differ in length"

runner = ModelRunner(args.model)
diffs, same = [], 0
for i in range(args.n):
    name = rows.loc[i, "id_code"]
    img = cv2.imread(str(Path(args.ddr) / name))
    if img is None:
        print(f"skip {name}: not found")
        continue
    p = runner.run(img).probs
    diff = float(np.abs(p - stored[i]).max())
    diffs.append(diff)
    same += int(p.argmax() == stored[i].argmax())
    print(f"{name:28s} app {C.LABELS[p.argmax()]:16s} notebook "
          f"{C.LABELS[stored[i].argmax()]:16s} max diff {diff:.4f}")

if diffs:
    print(f"\n{args.model}: {len(diffs)} images | same grade {same} | "
          f"mean max diff {np.mean(diffs):.4f} | worst {max(diffs):.4f}")

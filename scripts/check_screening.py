"""Optional: does the app reproduce the notebook on real test images?

    python scripts/check_screening.py --ddr "<folder with raw DDR jpgs>" --n 10

Runs the app's engine on the first N DDR test images (v2 split order) and compares with the
notebook's stored probs_test.npy. Expect small differences: the notebook read JPEG-cached
images on a GPU in float16; the app reads raw images on CPU in float32. What must match is
the decisions: grade and referral.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2           # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C                  # noqa: E402
from src import registry                     # noqa: E402
from src.decisions import decide             # noqa: E402
from src.inference import ScreeningEngine    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ddr", required=True, help="folder holding the raw DDR images")
ap.add_argument("--n", type=int, default=10)
args = ap.parse_args()

split = pd.read_csv(C.SPLIT_FILES["v2"])
test = split[split["split"] == "test"].reset_index(drop=True)
stored = np.load(registry.model_dir(C.SCREENING_MODEL_ID) / "probs_test.npy")
assert len(test) == len(stored), "split rows and stored probabilities differ in length"

engine = ScreeningEngine()
diffs, same_grade, same_ref = [], 0, 0
for i in range(args.n):
    name = test.loc[i, "id_code"]
    img = cv2.imread(str(Path(args.ddr) / name))
    if img is None:
        print(f"skip {name}: not found")
        continue
    r = engine.run(img)
    want = decide(stored[i], engine.thresholds)
    diff = float(np.abs(r.probs - stored[i]).max())
    diffs.append(diff)
    same_grade += r.decision.grade == want.grade
    same_ref += r.decision.refer == want.refer
    print(f"{name:28s} app {C.LABELS[r.decision.grade]:16s} notebook {C.LABELS[want.grade]:16s}"
          f" max diff {diff:.4f}")

if diffs:
    print(f"\n{len(diffs)} images | same grade {same_grade} | same referral {same_ref} | "
          f"mean max diff {np.mean(diffs):.4f} | worst {max(diffs):.4f}")

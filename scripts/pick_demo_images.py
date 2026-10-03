"""Copy the images used in the video demo into one folder, named in upload order.

    python scripts/pick_demo_images.py --ddr "<folder with raw DDR jpgs>"
    python scripts/pick_demo_images.py --ddr "<DDR folder>" --idrid "<IDRiD folder>"

DDR images are chosen from the screening model's stored test predictions, so no model runs
to choose them. The IDRiD image is chosen by running the app's model on a few Severe-graded
IDRiD photographs, because the notebook did not save which file each IDRiD row came from.
With --verify (the default) the app's model is also run on every chosen DDR image, to confirm
the app gives the same grade and referral the notebook stored.

Writes demo_images/ with the images and expected_outputs.csv.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from src import config as C
from src import data
from src.runner import ModelRunner

GRADE_NAME = re.compile(r"^IDRiD_(\d{3})(test)?\.jpg$", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--ddr", required=True, type=Path, help="folder holding the raw DDR images")
ap.add_argument("--idrid", type=Path, help="folder holding the IDRiD download (optional)")
ap.add_argument("--labels", type=Path, help="IDRiD rehost label CSV, used if no official CSV")
ap.add_argument("--idrid-tries", type=int, default=12, help="Severe IDRiD images to run")
ap.add_argument("--out", type=Path, default=Path("demo_images"))
ap.add_argument("--no-verify", action="store_true", help="skip running the model on DDR picks")
args = ap.parse_args()

model_id = C.SCREENING_MODEL_ID
res = data.run_results(model_id)
ref_t, abs_t = float(res["referral_threshold"]), float(res["abstention_threshold"])

split = pd.read_csv(C.SPLIT_FILES["v2"])
test = split[split["split"] == "test"].reset_index(drop=True)
p = data.probs(model_id)
y = test["diagnosis"].to_numpy()
assert len(p) == len(test), "stored test probabilities and split rows differ in length"

pred, conf, refp = p.argmax(1), p.max(1), p[:, 2:].sum(1)
refer, graded = refp >= ref_t, conf >= abs_t


def best(mask, score):
    idx = np.where(mask)[0]
    return int(idx[np.argmax(score[idx])]) if len(idx) else None


picks = {
    "1_no_dr": best((y == 0) & (pred == 0) & graded, conf),
    "2_mild_but_referred": best((pred == 1) & refer & graded & (y >= 2), conf),
    "3_not_graded": best(~graded & refer & (y >= 2), -conf),
    "5_severe": best((y == 3) & (pred == 3) & graded, conf),
}


def screen_text(c, r, k):
    return ("Not graded" if c < abs_t else C.LABELS[k]), ("Refer" if r >= ref_t else "No referral")


args.out.mkdir(parents=True, exist_ok=True)
rows = []
for name, i in picks.items():
    if i is None:
        print(f"{name}: no test image matches, skipped")
        continue
    src = args.ddr / test.loc[i, "id_code"]
    if not src.exists():
        print(f"{name}: {src.name} not found in {args.ddr}, skipped")
        continue
    shutil.copy(src, args.out / f"{name}{src.suffix.lower()}")
    grade_shown, referral = screen_text(conf[i], refp[i], pred[i])
    rows.append({"file": f"{name}{src.suffix.lower()}", "source": src.name, "set": "DDR test",
                 "true grade": C.LABELS[y[i]], "app shows": grade_shown, "referral": referral,
                 "confidence": round(float(conf[i]), 4),
                 "referable prob": round(float(refp[i]), 4), "_i": i})

runner = None
if not args.no_verify and rows:
    runner = ModelRunner(model_id)
    print("\nChecking the app's model against the stored predictions:")
    for r in rows:
        img = cv2.imread(str(args.out / r["file"]))
        q = runner.run(img).probs
        i = r.pop("_i")
        same_grade = q.argmax() == p[i].argmax()
        same_ref = (q[2:].sum() >= ref_t) == refer[i]
        same_abs = (q.max() >= abs_t) == graded[i]
        ok = "OK" if same_grade and same_ref and same_abs else "DIFFERS"
        print(f"  {r['file']:28s} {ok}  app conf {q.max():.3f} vs stored {conf[i]:.3f}, "
              f"app referable {q[2:].sum():.3f} vs stored {refp[i]:.3f}")
else:
    for r in rows:
        r.pop("_i")


def idrid_labels(root: Path, extra: Path | None) -> dict:
    frames = []
    for f in root.rglob("*.csv"):
        if "grading" in f.name.lower() and "label" in f.name.lower():
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            if {"Image name", "Retinopathy grade"} <= set(df.columns):
                df["testing"] = "testing" in f.name.lower()
                frames.append(df.rename(columns={"Image name": "id_code",
                                                 "Retinopathy grade": "diagnosis"}))
    if not frames and extra is not None:
        df = pd.read_csv(extra)
        df["testing"] = False
        frames = [df]
    if not frames:
        return {}
    df = pd.concat(frames).dropna(subset=["id_code", "diagnosis"])
    out = {}
    for name, g, t in zip(df["id_code"], df["diagnosis"], df["testing"]):
        m = re.match(r"^IDRiD_(\d{3})(test)?$", str(name).strip(), re.I)
        if m:
            out[f"{'test' if t or m.group(2) else 'train'}/{m.group(1)}"] = int(g)
    return out


if args.idrid is not None:
    labels = idrid_labels(args.idrid, args.labels)
    if not labels:
        print("\nIDRiD: no grading CSV found; pass --labels to the rehost CSV. Skipped.")
    else:
        images = {}
        for f in args.idrid.rglob("*.jpg"):
            m = GRADE_NAME.match(f.name)
            if m:
                testing = bool(m.group(2)) or any("testing" in s.lower() for s in f.parts)
                images[f"{'test' if testing else 'train'}/{m.group(1)}"] = f
        severe = sorted(k for k, g in labels.items() if g == 3 and k in images)
        if not severe:
            print("\nIDRiD: no Severe-graded image found. Skipped.")
        else:
            runner = runner or ModelRunner(model_id)
            print(f"\nRunning the model on {min(len(severe), args.idrid_tries)} Severe IDRiD images:")
            tried = []
            for k in severe[:args.idrid_tries]:
                q = runner.run(cv2.imread(str(images[k]))).probs
                tried.append((k, q))
                print(f"  {images[k].name:20s} top {C.LABELS[q.argmax()]:16s} conf {q.max():.3f}")
            correct = [(k, q) for k, q in tried if q.argmax() == 3 and q.max() >= abs_t]
            k, q = max(correct or tried, key=lambda kq: kq[1].max())
            shutil.copy(images[k], args.out / "4_idrid_severe.jpg")
            grade_shown, referral = screen_text(q.max(), q[2:].sum(), q.argmax())
            rows.append({"file": "4_idrid_severe.jpg", "source": images[k].name, "set": "IDRiD",
                         "true grade": C.LABELS[3], "app shows": grade_shown,
                         "referral": referral, "confidence": round(float(q.max()), 4),
                         "referable prob": round(float(q[2:].sum()), 4)})
            if not correct:
                print("  none was graded Severe with enough confidence; kept the most confident")

table = pd.DataFrame(rows).sort_values("file")
table.to_csv(args.out / "expected_outputs.csv", index=False)
print(f"\nThresholds: referral {ref_t:.2f}, abstention {abs_t:.4f}\n")
print(table.to_string(index=False))
print(f"\nSaved to {args.out.resolve()}")
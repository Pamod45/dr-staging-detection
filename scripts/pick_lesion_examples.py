"""Find the DR grade of each IDRiD segmentation image, and copy one expert-annotated image per
grade into content/lesions/grade_N/ for the Home page.

The segmentation images (IDRiD_01 to IDRiD_81) and the grading images (IDRiD_001 to IDRiD_516)
are numbered separately and ship no mapping, so grades are found by matching picture content:
both sides go through the app's own retina crop, shrunk to 128 px grey, and compared by
correlation. A match needs correlation of at least --min-corr; anything weaker is reported as
unmatched rather than guessed.

    python scripts/pick_lesion_examples.py --idrid "C:/.../Datasets/IDiRD"
    python scripts/pick_lesion_examples.py --idrid "C:/.../Datasets/IDiRD" --copy

--idrid is searched recursively, so the official folder layout ("A. Segmentation",
"B. Disease Grading") or a flat rehost both work. Grades come from the official grading CSVs
when present (the Kaggle rehost has 27 known label errors), otherwise from --labels.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import preprocess  

LESIONS = ("MA", "HE", "EX", "SE")
SEG_NAME = re.compile(r"^IDRiD_(\d{2})\.jpg$", re.I)
GRADE_NAME = re.compile(r"^IDRiD_(\d{3})(test)?\.jpg$", re.I)
LABELS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}


def grading_key(name: str, testing: bool) -> str:
    """The official training and testing sets both start at IDRiD_001, so a grading image is
    identified by split and number together. The Kaggle rehost marks test images 'IDRiD_001test'."""
    m = re.match(r"^IDRiD_(\d{3})(test)?$", name.strip(), re.I)
    return f"{'test' if testing or m.group(2) else 'train'}/{m.group(1)}" if m else name


def image_is_testing(path: Path) -> bool:
    return path.stem.lower().endswith("test") or any("testing" in p.lower() for p in path.parts)


def signature(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    out, _ = preprocess(img, out=128)
    if out is None:
        return None
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


def load_labels(root: Path, extra: Path | None) -> tuple[dict, str]:
    official = [p for p in root.rglob("*.csv") if "grading" in p.name.lower()
                and "label" in p.name.lower()]
    frames, source = [], ""
    for p in official:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        if {"Image name", "Retinopathy grade"} <= set(df.columns):
            df = df.rename(columns={"Image name": "id_code", "Retinopathy grade": "diagnosis"})
            df["testing"] = "testing" in p.name.lower()
            frames.append(df)
    if frames:
        source = "official IDRiD grading CSVs"
    elif extra is not None:
        df = pd.read_csv(extra)
        df["testing"] = False
        frames = [df]
        source = f"{extra.name} (rehost labels; 27 are known to differ from the official ones)"
    else:
        sys.exit("No grading labels found. Pass --labels path/to/idrid_labels.csv")
    df = pd.concat(frames)
    df = df.dropna(subset=["id_code", "diagnosis"])
    return {grading_key(str(i), bool(t)): int(g)
            for i, g, t in zip(df["id_code"], df["diagnosis"], df["testing"])}, source


def mask_files(root: Path, stem: str) -> dict:
    out = {}
    for code in LESIONS + ("OD",):
        hits = list(root.rglob(f"{stem}_{code}.tif"))
        if hits:
            out[code] = hits[0]
    return out


def lesion_area(path: Path) -> int:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return 0 if m is None else int((m > 0).sum())


ap = argparse.ArgumentParser()
ap.add_argument("--idrid", required=True, type=Path, help="folder holding the IDRiD download")
ap.add_argument("--labels", type=Path, help="grading CSV if the official ones are not there")
ap.add_argument("--min-corr", type=float, default=0.95)
ap.add_argument("--copy", action="store_true", help="copy one image per grade into content/")
args = ap.parse_args()

labels, source = load_labels(args.idrid, args.labels)
seg = sorted(p for p in args.idrid.rglob("*.jpg") if SEG_NAME.match(p.name))
grading = sorted(p for p in args.idrid.rglob("*.jpg") if GRADE_NAME.match(p.name))
print(f"{len(seg)} segmentation images, {len(grading)} grading images, labels from {source}")
if not seg or not grading:
    sys.exit("Need both the segmentation images and the grading images under --idrid.")

print("computing signatures ...")
g_names, g_sigs = [], []
for p in grading:
    s = signature(p)
    if s is not None:
        g_names.append(grading_key(p.stem, image_is_testing(p)))
        g_sigs.append(s)
g_sigs = np.stack(g_sigs)

rows = []
for p in seg:
    s = signature(p)
    if s is None:
        continue
    corr = g_sigs @ s / len(s)
    best = int(corr.argmax())
    matched = corr[best] >= args.min_corr
    masks = mask_files(args.idrid, p.stem)
    rows.append({
        "seg_image": p.stem,
        "grading_image": g_names[best].replace("/", " IDRiD_") if matched else "",
        "corr": round(float(corr[best]), 4),
        "grade": labels.get(g_names[best]) if matched else None,
        "lesion_types": sum(c in masks for c in LESIONS),
        "lesion_area": sum(lesion_area(masks[c]) for c in LESIONS if c in masks),
        "path": p, "masks": masks,
    })

table = pd.DataFrame(rows)
print(table.drop(columns=["path", "masks"]).to_string(index=False))
found = table.dropna(subset=["grade"])
print(f"\nmatched {len(found)} of {len(table)}; per grade:",
      found["grade"].astype(int).value_counts().sort_index().to_dict())

picks = {}
for g in (1, 2, 3, 4):
    cand = found[found["grade"] == g]
    if cand.empty:
        print(f"grade {g} ({LABELS[g]}): no annotated image")
        continue
    # most lesion types first; Mild favours the least lesion area (a clean example),
    # the other grades the most (clearly visible signs)
    cand = cand.sort_values(["lesion_types", "lesion_area"],
                            ascending=[False, g == 1])
    picks[g] = cand.iloc[0]
    r = picks[g]
    print(f"grade {g} ({LABELS[g]}): {r['seg_image']} = {r['grading_image']} "
          f"(corr {r['corr']}), {r['lesion_types']} lesion types")

if args.copy:
    for g, r in picks.items():
        dest = ROOT / "content" / "lesions" / f"grade_{g}"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        shutil.copy2(r["path"], dest / r["path"].name)
        for m in r["masks"].values():
            shutil.copy2(m, dest / m.name)
        print(f"copied {r['seg_image']} and {len(r['masks'])} masks to {dest.relative_to(ROOT)}")
elif picks:
    print("\ndry run: add --copy to copy these into content/lesions/grade_N/")

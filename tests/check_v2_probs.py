import numpy as np, json
from pathlib import Path

ROOT = Path(".")
y = np.load(ROOT / "models/_shared/test_labels_v2.npy")
print("v2 test rows:", len(y))

for name in ["v2_512_2000", "v2_512_full", "v2_512_merged",
             "v2_768_best", "v2_768_qwk"]:
    d = ROOT / "models" / name
    f = d / "probs_test.npy"
    if not f.exists():
        print(f"{name:16s} no probs_test.npy (weights only)")
        continue
    p = np.load(f)
    acc = (p.argmax(1) == y).mean()
    res = json.load(open(d / "results.json"))
    print(f"{name:16s} shape={p.shape} sums_to_1={np.allclose(p.sum(1),1)} "
          f"acc={acc:.4f}")
    print("   results.json keys:", list(res)[:8])

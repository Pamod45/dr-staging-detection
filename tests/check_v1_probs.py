import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(".")
SHARED = ROOT / "models/_shared"

# --- build the label arrays from the split files -------------------
v1 = pd.read_csv(SHARED / "ddr_splits.csv")
v2 = pd.read_csv(SHARED / "ddr_splits_7x6.csv")

y_val_v1 = v1.loc[v1["split"] == "val", "diagnosis"].to_numpy()
np.save(SHARED / "val_labels_v1.npy", y_val_v1)
np.save(SHARED / "test_labels_v1.npy",
        v1.loc[v1["split"] == "test", "diagnosis"].to_numpy())
np.save(SHARED / "test_labels_v2.npy",
        v2.loc[v2["split"] == "test", "diagnosis"].to_numpy())
print("val rows v1:", len(y_val_v1))          # expect 1792

# --- check every v1 folder ----------------------------------------
expected = {                                   # peak val QWK from the notebook tables
    "v1_res_224": 0.8178, "v1_res_380": 0.8478, "v1_res_512": 0.8603,
    "v1_oversampled_512": 0.8601, "v1_bal_focal_512": 0.8769,
    "v1_focal_512": 0.8682, "v1_focal_alpha_512": 0.8522,
    "v1_focal_clahe_512": 0.8370,
}

for name, qwk in expected.items():
    d = ROOT / "models" / name
    p = np.load(d / "val_probs.npy")
    h = pd.read_json(d / "history.json")
    acc = (p.argmax(1) == y_val_v1).mean()
    print(f"{name:22s} shape={p.shape} sums_to_1={np.allclose(p.sum(1),1)} "
          f"acc={acc:.4f} peak_qwk={h['val_qwk'].max():.4f} (expect ~{qwk})")
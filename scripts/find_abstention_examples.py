import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
run = Path("models/v2_768_best")
res = json.loads((run / "results.json").read_text())
t, ref_t = res["abstention_threshold"], res["referral_threshold"]

split = pd.read_csv("models/_shared/ddr_splits_7x6.csv")
test = split[split["split"] == "test"].reset_index(drop=True)
p = np.load(run / "probs_test.npy")

conf, ref = p.max(1), p[:, 2:].sum(1)
rows = [{"image": test.loc[i, "id_code"], "true grade": LABELS[test.loc[i, "diagnosis"]],
         "top guess": LABELS[p[i].argmax()], "top prob": round(float(conf[i]), 3),
         "referred": bool(ref[i] >= ref_t)}
        for i in np.where(conf < t)[0]]
df = pd.DataFrame(rows).sort_values("top prob")
print(f"{len(df)} test images withheld below {t:.4f}\n")
print(df.head(25).to_string(index=False))
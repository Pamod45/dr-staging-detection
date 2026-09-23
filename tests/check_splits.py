import pandas as pd, numpy as np

v1 = pd.read_csv("notebooks/dr_grading_kaggle_v1/artefacts/ddr_splits.csv")
v2 = pd.read_csv("notebooks/dr_grading_kaggle_v2_minimal/artefacts/shared/ddr_splits_7x6.csv")

print("v1 columns:", list(v1.columns))
print("v2 columns:", list(v2.columns))
print("v1 split counts:\n", v1["split"].value_counts())
print("v2 split counts:\n", v2["split"].value_counts())

ID, SPLIT = "id_code", "split"

t1 = v1.loc[v1[SPLIT] == "test", ID].astype(str)
t2 = v2.loc[v2[SPLIT] == "test", ID].astype(str)

print("test sizes:", len(t1), len(t2))
print("same images, ignoring order:", set(t1) == set(t2))
print("same images, same order:", list(t1) == list(t2))
print("in v1 only:", len(set(t1) - set(t2)), " in v2 only:", len(set(t2) - set(t1)))

y1 = np.load("notebooks/dr_grading_kaggle_v1/artefacts/test/test_labels.npy")
print("v1 labels length matches v1 test rows:", len(y1) == len(t1))
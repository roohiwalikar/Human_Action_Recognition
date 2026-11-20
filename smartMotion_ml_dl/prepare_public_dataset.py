import os
import pandas as pd
import numpy as np

PUBLIC_FILE = "train.parquet"
OUTPUT_FOLDER = "public_dataset"

TARGET_LABELS = [
    "waving",
    "reading",
    "dancing",
    "lying_down"
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading public dataset...")

df = pd.read_parquet(PUBLIC_FILE)

print("Total public rows:", len(df))
print("Available labels:")
print(df["label"].value_counts())

feature_cols = [c for c in df.columns if c != "label"]

df[feature_cols] = df[feature_cols].apply(
    pd.to_numeric,
    errors="coerce"
)

df[feature_cols] = (
    df[feature_cols]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

print()
print("Extracting required public classes...")

for label in TARGET_LABELS:

    part = df[df["label"] == label].copy()

    if part.empty:
        print(label, ": NOT FOUND")
        continue

    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"{label}.csv"
    )

    part.to_csv(
        output_file,
        index=False
    )

    print(
        label,
        ":",
        len(part),
        "frames saved"
    )

print()
print("====================================")
print("PUBLIC DATASET PREPARATION COMPLETE")
print("====================================")
print("Folder:", OUTPUT_FOLDER)
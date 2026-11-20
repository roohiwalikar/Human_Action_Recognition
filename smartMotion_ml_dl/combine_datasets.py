import os
import pandas as pd

# ==========================================
# HUMAN ACTION RECOGNITION
# COMBINE TWO DATASETS
# ==========================================

DATASET_1 = "dataset"
DATASET_2 = "test_dataset"
OUTPUT_FOLDER = "combined_dataset"

ACTIONS = [
    "standing",
    "sitting",
    "hands_up",
    "walking",
    "jumping",
    "waving"
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("==========================================")
print(" COMBINING DATASETS")
print("==========================================")

for action in ACTIONS:

    file1 = os.path.join(DATASET_1, action + ".csv")
    file2 = os.path.join(DATASET_2, action + ".csv")
    output_file = os.path.join(OUTPUT_FOLDER, action + ".csv")

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    combined = pd.concat([df1, df2], ignore_index=True)

    combined.to_csv(output_file, index=False)

    print(
        action,
        "->",
        len(df1),
        "+",
        len(df2),
        "=",
        len(combined),
        "frames"
    )

print()
print("==========================================")
print(" DATASETS COMBINED SUCCESSFULLY")
print("==========================================")
print("Saved in:", OUTPUT_FOLDER)
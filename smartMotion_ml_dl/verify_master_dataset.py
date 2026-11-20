import os
import pandas as pd
import numpy as np

# ============================================================
# 9-CLASS MASTER DATASET VERIFICATION
# ============================================================

TRAIN_FOLDER = "master_dataset_train"
TEST_FOLDER = "master_dataset_test"

CLASSES = [
    "standing",
    "sitting",
    "hands_up",
    "walking",
    "jumping",
    "waving",
    "reading",
    "dancing",
    "lying_down"
]

EXPECTED_FEATURES = 132


def verify_file(file_path, expected_class):
    """Verify one CSV file."""

    if not os.path.exists(file_path):
        print(f"❌ MISSING: {file_path}")
        return False

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ ERROR reading {file_path}: {e}")
        return False

    # Check label column
    if "label" not in df.columns:
        print(f"❌ {file_path}: label column missing")
        return False

    # Check label values
    labels = df["label"].dropna().unique()

    if len(labels) != 1 or labels[0] != expected_class:
        print(
            f"❌ {file_path}: incorrect label(s): {labels}"
        )
        return False

    # Count feature columns
    feature_cols = [
        c for c in df.columns
        if c != "label"
    ]

    if len(feature_cols) != EXPECTED_FEATURES:
        print(
            f"❌ {file_path}: "
            f"expected {EXPECTED_FEATURES} features, "
            f"found {len(feature_cols)}"
        )
        return False

    # Check numeric features
    numeric_df = df[feature_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Check NaN
    nan_count = numeric_df.isna().sum().sum()

    if nan_count > 0:
        print(
            f"❌ {file_path}: "
            f"{nan_count} NaN values found"
        )
        return False

    # Check infinity
    inf_count = np.isinf(
        numeric_df.to_numpy()
    ).sum()

    if inf_count > 0:
        print(
            f"❌ {file_path}: "
            f"{inf_count} infinity values found"
        )
        return False

    print(
        f"✅ {expected_class:12s} | "
        f"{len(df):4d} rows | "
        f"{len(feature_cols):3d} features"
    )

    return True


print()
print("=" * 70)
print("9-CLASS MASTER DATASET VERIFICATION")
print("=" * 70)
print()

all_good = True

# ============================================================
# VERIFY TRAINING DATA
# ============================================================

print("TRAINING DATA")
print("-" * 70)

for action in CLASSES:

    file_path = os.path.join(
        TRAIN_FOLDER,
        action + ".csv"
    )

    result = verify_file(
        file_path,
        action
    )

    if not result:
        all_good = False

print()

# ============================================================
# VERIFY TEST DATA
# ============================================================

print("TEST DATA")
print("-" * 70)

for action in CLASSES:

    file_path = os.path.join(
        TEST_FOLDER,
        action + ".csv"
    )

    result = verify_file(
        file_path,
        action
    )

    if not result:
        all_good = False

print()

# ============================================================
# CHECK CLASS BALANCE
# ============================================================

print("=" * 70)
print("CLASS SUMMARY")
print("=" * 70)

train_total = 0
test_total = 0

for action in CLASSES:

    train_file = os.path.join(
        TRAIN_FOLDER,
        action + ".csv"
    )

    test_file = os.path.join(
        TEST_FOLDER,
        action + ".csv"
    )

    train_count = (
        len(pd.read_csv(train_file))
        if os.path.exists(train_file)
        else 0
    )

    test_count = (
        len(pd.read_csv(test_file))
        if os.path.exists(test_file)
        else 0
    )

    train_total += train_count
    test_total += test_count

    print(
        f"{action:12s} -> "
        f"TRAIN: {train_count:4d} | "
        f"TEST: {test_count:4d}"
    )

print("-" * 70)

print(
    f"TOTAL        -> "
    f"TRAIN: {train_total:4d} | "
    f"TEST: {test_total:4d}"
)

print()

# ============================================================
# FINAL RESULT
# ============================================================

print("=" * 70)

if all_good:
    print("✅ DATASET VERIFICATION PASSED")
    print("All 9 classes have valid 132-feature pose data.")
else:
    print("❌ DATASET VERIFICATION FAILED")
    print("Please fix the errors shown above.")

print("=" * 70)
print()
import os
import pandas as pd
import numpy as np

# ============================================================
# 9-CLASS MASTER DATASET CREATION
# ============================================================

OWN_TRAIN_FOLDERS = [
    "dataset",
    "new_dataset_train"
]

OWN_TEST_FOLDERS = [
    "test_dataset",
    "new_dataset_test"
]

PUBLIC = "public_dataset"

MASTER_TRAIN = "master_dataset_train"
MASTER_TEST = "master_dataset_test"

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

PUBLIC_CLASSES = [
    "waving",
    "reading",
    "dancing",
    "lying_down"
]

EXPECTED_FEATURES = 132

# Create output folders if they don't exist
os.makedirs(MASTER_TRAIN, exist_ok=True)
os.makedirs(MASTER_TEST, exist_ok=True)


# ============================================================
# READ AND STANDARDIZE DATA
# ============================================================

def prepare_dataframe(file_path, action):

    df = pd.read_csv(file_path)

    # --------------------------------------------------------
    # Separate label column if present
    # --------------------------------------------------------

    if "label" in df.columns:
        feature_df = df.drop(columns=["label"]).copy()
    else:
        feature_df = df.copy()

    # --------------------------------------------------------
    # Keep only numeric pose data
    # --------------------------------------------------------

    feature_df = feature_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        0
    ).fillna(0)

    # --------------------------------------------------------
    # Check feature count
    # --------------------------------------------------------

    if feature_df.shape[1] < EXPECTED_FEATURES:

        print(
            f"ERROR: {file_path} has only "
            f"{feature_df.shape[1]} features."
        )

        return None

    # --------------------------------------------------------
    # Keep exactly 132 pose features
    # --------------------------------------------------------

    feature_df = feature_df.iloc[
        :, :EXPECTED_FEATURES
    ].copy()

    # --------------------------------------------------------
    # Give every dataset the same feature names
    # --------------------------------------------------------

    feature_df.columns = [
        f"feature_{i}"
        for i in range(EXPECTED_FEATURES)
    ]

    # --------------------------------------------------------
    # Add correct class label
    # --------------------------------------------------------

    feature_df["label"] = action

    return feature_df


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("CREATING 9-CLASS MASTER DATASET")
print("=" * 70)
print()


total_train = 0
total_test = 0


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for action in CLASSES:

    train_parts = []
    test_parts = []

    print()
    print("-" * 70)
    print(f"PROCESSING: {action.upper()}")
    print("-" * 70)

    # ========================================================
    # OWN TRAINING DATA
    # ========================================================

    for folder in OWN_TRAIN_FOLDERS:

        file_path = os.path.join(
            folder,
            action + ".csv"
        )

        if os.path.exists(file_path):

            df = prepare_dataframe(
                file_path,
                action
            )

            if df is not None:

                train_parts.append(df)

                print(
                    f"Own TRAIN  ({folder}) -> "
                    f"{len(df)} rows"
                )

        else:

            print(
                f"WARNING: Missing TRAIN file -> "
                f"{file_path}"
            )

    # ========================================================
    # OWN TESTING DATA
    # ========================================================

    for folder in OWN_TEST_FOLDERS:

        file_path = os.path.join(
            folder,
            action + ".csv"
        )

        if os.path.exists(file_path):

            df = prepare_dataframe(
                file_path,
                action
            )

            if df is not None:

                test_parts.append(df)

                print(
                    f"Own TEST   ({folder}) -> "
                    f"{len(df)} rows"
                )

        else:

            print(
                f"WARNING: Missing TEST file -> "
                f"{file_path}"
            )

    # ========================================================
    # PUBLIC DATA
    # USED ONLY FOR TRAINING AUGMENTATION
    # ========================================================

    public_file = os.path.join(
        PUBLIC,
        action + ".csv"
    )

    if action in PUBLIC_CLASSES and os.path.exists(public_file):

        public_df = prepare_dataframe(
            public_file,
            action
        )

        if public_df is not None:

            # Shuffle reproducibly
            public_df = public_df.sample(
                frac=1,
                random_state=42
            ).reset_index(drop=True)

            # Add ALL public data to TRAINING
            train_parts.append(
                public_df
            )

            print(
                f"Public TRAIN augmentation -> "
                f"{len(public_df)} rows"
            )

            print(
                "Public TEST -> NOT USED"
            )

    elif action in PUBLIC_CLASSES:

        print(
            f"WARNING: Public file not found -> "
            f"{public_file}"
        )

    # ========================================================
    # SAVE TRAIN DATA
    # ========================================================

    if train_parts:

        final_train = pd.concat(
            train_parts,
            ignore_index=True
        )

        # Shuffle final training dataset
        final_train = final_train.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

        output_file = os.path.join(
            MASTER_TRAIN,
            action + ".csv"
        )

        final_train.to_csv(
            output_file,
            index=False
        )

        print(
            f"FINAL TRAIN -> "
            f"{len(final_train)} rows | "
            f"{len(final_train.columns) - 1} features"
        )

        total_train += len(final_train)

    else:

        print(
            f"ERROR: No training data found for {action}"
        )

    # ========================================================
    # SAVE TEST DATA
    # ========================================================

    if test_parts:

        final_test = pd.concat(
            test_parts,
            ignore_index=True
        )

        # Shuffle final test dataset
        final_test = final_test.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

        output_file = os.path.join(
            MASTER_TEST,
            action + ".csv"
        )

        final_test.to_csv(
            output_file,
            index=False
        )

        print(
            f"FINAL TEST  -> "
            f"{len(final_test)} rows | "
            f"{len(final_test.columns) - 1} features"
        )

        total_test += len(final_test)

    else:

        print(
            f"ERROR: No test data found for {action}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("9-CLASS MASTER DATASET SUMMARY")
print("=" * 70)

print()

print(
    f"{'CLASS':15s} | "
    f"{'TRAIN':>7s} | "
    f"{'TEST':>7s} | "
    f"{'FEATURES':>8s}"
)

print("-" * 55)


for action in CLASSES:

    train_file = os.path.join(
        MASTER_TRAIN,
        action + ".csv"
    )

    test_file = os.path.join(
        MASTER_TEST,
        action + ".csv"
    )

    train_count = 0
    test_count = 0
    features = 0

    if os.path.exists(train_file):

        train_df = pd.read_csv(
            train_file
        )

        train_count = len(train_df)
        features = len(train_df.columns) - 1

    if os.path.exists(test_file):

        test_df = pd.read_csv(
            test_file
        )

        test_count = len(test_df)

    print(
        f"{action:15s} | "
        f"{train_count:7d} | "
        f"{test_count:7d} | "
        f"{features:8d}"
    )


print("-" * 55)

print(
    f"{'TOTAL':15s} | "
    f"{total_train:7d} | "
    f"{total_test:7d} | "
    f"{EXPECTED_FEATURES:8d}"
)

print()

print("=" * 70)
print("9-CLASS MASTER DATASET READY")
print("=" * 70)

print()
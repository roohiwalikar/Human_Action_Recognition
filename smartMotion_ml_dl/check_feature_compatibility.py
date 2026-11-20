import pandas as pd
import os

OWN_FILE = os.path.join(
    "dataset",
    "standing.csv"
)

PUBLIC_FILE = os.path.join(
    "public_dataset",
    "waving.csv"
)

print("=" * 70)
print("CHECKING FEATURE COMPATIBILITY")
print("=" * 70)

# ------------------------------------------------------------
# LOAD FILES
# ------------------------------------------------------------

own = pd.read_csv(OWN_FILE)
public = pd.read_csv(PUBLIC_FILE)

print()
print("OWN DATASET")
print("-" * 70)

print("Shape:", own.shape)
print("Columns:")
print(list(own.columns))

print()
print("PUBLIC DATASET")
print("-" * 70)

print("Shape:", public.shape)
print("Columns:")
print(list(public.columns))

# ------------------------------------------------------------
# REMOVE LABEL
# ------------------------------------------------------------

if "label" in own.columns:
    own_features = own.drop(columns=["label"])
else:
    own_features = own

if "label" in public.columns:
    public_features = public.drop(columns=["label"])
else:
    public_features = public

# ------------------------------------------------------------
# COMPARE NUMBER OF FEATURES
# ------------------------------------------------------------

print()
print("=" * 70)
print("FEATURE COUNT")
print("=" * 70)

print("Own features   :", own_features.shape[1])
print("Public features:", public_features.shape[1])

# ------------------------------------------------------------
# COMPARE COLUMN NAMES
# ------------------------------------------------------------

print()
print("=" * 70)
print("FIRST 20 OWN COLUMNS")
print("=" * 70)

for i, col in enumerate(own_features.columns[:20]):
    print(i, "->", col)

print()
print("=" * 70)
print("FIRST 20 PUBLIC COLUMNS")
print("=" * 70)

for i, col in enumerate(public_features.columns[:20]):
    print(i, "->", col)

# ------------------------------------------------------------
# COMPARE DATA TYPES
# ------------------------------------------------------------

print()
print("=" * 70)
print("DATA TYPES")
print("=" * 70)

print("OWN:")
print(own_features.dtypes.head(20))

print()
print("PUBLIC:")
print(public_features.dtypes.head(20))

# ------------------------------------------------------------
# BASIC VALUE CHECK
# ------------------------------------------------------------

print()
print("=" * 70)
print("FIRST ROW VALUES")
print("=" * 70)

print()
print("OWN FIRST ROW:")
print(own_features.iloc[0].head(20).to_string())

print()
print("PUBLIC FIRST ROW:")
print(public_features.iloc[0].head(20).to_string())

print()
print("=" * 70)
print("CHECK COMPLETE")
print("=" * 70)
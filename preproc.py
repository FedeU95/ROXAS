import pandas as pd
from pathlib import Path

# --- Inputs ---
DATA_PATH        = "data/MSA.csv"           # mixed benign+malicious with labels
NEW_BENIGN_PATH  = "data/nas_data_processed_all_features.csv"    # new benign, no labels

# --- Outputs ---
OUT_DIR          = "data"
BENIGN_OUT       = f"{OUT_DIR}/benign.csv"
MALICIOUS_OUT    = f"{OUT_DIR}/malicious.csv"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Load ----------
df = pd.read_csv(DATA_PATH, low_memory=False)
new_benign = pd.read_csv(NEW_BENIGN_PATH, low_memory=False)

# ---------- 1) Separate benign vs malicious ----------
required = ["Binary_Label", "Multiclass_Label"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Required column missing in {DATA_PATH}: {col}")

benign_old = df[df["Binary_Label"] == 0].copy()
malicious  = df[df["Binary_Label"] == 1].copy()

# ---------- 2) Remove labels from BENIGN ----------
label_cols = ["Binary_Label", "Multiclass_Label"]
benign_old_features = benign_old.drop(columns=[c for c in label_cols if c in benign_old.columns])

# Concatenate old  + new benign 
benign_merged = pd.concat([benign_old_features, new_benign], axis=0, ignore_index=True)


malicious_final = malicious.copy() 

# ---------- 5) Save ----------
benign_merged.to_csv(BENIGN_OUT, index=False)
malicious_final.to_csv(MALICIOUS_OUT, index=False)

print(f"Benign saved:    {BENIGN_OUT}  shape={benign_merged.shape}")
print(f"Malicious saved: {MALICIOUS_OUT}  shape={malicious_final.shape}")


import pandas as pd
from pathlib import Path

# --- Inputs ---
DATA_PATH        = "data/msa_rrc.csv"              # mixed benign+malicious with label column 'label'
NEW_BENIGN_PATH  = "data/rrc_data_processed.csv"   # new benign, no labels

# --- Outputs ---
OUT_DIR          = "data"
BENIGN_OUT       = f"{OUT_DIR}/benign_rrc.csv"
MALICIOUS_OUT    = f"{OUT_DIR}/malicious_rrc.csv"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Load ----------
df = pd.read_csv(DATA_PATH, low_memory=False)
new_benign = pd.read_csv(NEW_BENIGN_PATH, low_memory=False)

# Drop junk index cols if present
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
new_benign = new_benign.loc[:, ~new_benign.columns.str.match(r"^Unnamed")]

# ---------- 1) Separate benign vs malicious ----------
if "label" not in df.columns:
    raise ValueError(f"Required column missing in {DATA_PATH}: label")

benign_old = df[df["label"] == 0].copy()
malicious  = df[df["label"] != 0].copy()

# ---------- 2) Remove labels from BENIGN ----------
benign_old_features = benign_old.drop(columns=["label"], errors="ignore")
new_benign_features = new_benign.drop(columns=["label", "Binary_Label", "Multiclass_Label"], errors="ignore")

# ---------- 3) Feature intersection (present in ALL needed places) ----------
exclude = {"label", "Binary_Label", "Multiclass_Label"}

old_set = set(benign_old_features.columns) - exclude
new_set = set(new_benign_features.columns) - exclude
mal_set = set(malicious.columns) - exclude  # malicious still has 'label', excluded above

common_features = sorted(old_set & new_set & mal_set)
if not common_features:
    raise ValueError("No common feature columns found across (old benign) ∩ (new benign) ∩ (malicious).")

# ---------- 4) OPTIONAL: drop columns that are all-NaN in ANY split ----------
def keep_nonempty_cols(cols, *dfs):
    keep = []
    for c in cols:
        if all(not df_[c].isna().all() for df_ in dfs):
            keep.append(c)
    return keep

common_features = keep_nonempty_cols(common_features, benign_old_features, new_benign_features, malicious)

if not common_features:
    raise ValueError("After dropping all-NaN columns, no features remain.")

# ---------- 5) Build final datasets ----------
# Benign: features only, no labels
benign_merged = pd.concat(
    [
        benign_old_features.loc[:, common_features],
        new_benign_features.loc[:, common_features],
    ],
    axis=0,
    ignore_index=True
)

# Malicious: same features + labels
malicious_final = malicious.loc[:, common_features].copy()
malicious_final["Multiclass_Label"] = malicious["label"].to_numpy()
malicious_final["Binary_Label"] = 1  # optional but matches your NAS format

# ---------- Save ----------
benign_merged.to_csv(BENIGN_OUT, index=False)
malicious_final.to_csv(MALICIOUS_OUT, index=False)

print(f"Final common features: {len(common_features)}")
print(f"Benign saved:    {BENIGN_OUT}  shape={benign_merged.shape}")
print(f"Malicious saved: {MALICIOUS_OUT}  shape={malicious_final.shape}")

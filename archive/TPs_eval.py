import os, json, random
import numpy as np
import pandas as pd
import joblib
from vote import Ensemble

# ---- Config ----
MODEL_PATH   = "models/AD_MSA_nas_XGB_aug.pkl"      # model
SCALER_PATH  = "models/AD_MSA_scaler_aug.pkl"       # matching scaler
ART_PATH     = "models/AD_MSA_artifacts.json"       # selected_features, medians, threshold, target
OUTPUT_DIR   = "TPs_eva;"
TOTAL_TPS    = 100
SEED         = 0
random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load data ----
df = pd.read_csv("data/MSA.csv", low_memory=False)
df.drop(df.columns[0], axis=1, inplace=True)
df = df.drop_duplicates()
assert {"Binary_Label","Multiclass_Label"}.issubset(df.columns)

y_bin  = df["Binary_Label"].astype(int)
y_mult = df["Multiclass_Label"].astype(int)
X_all  = df.drop(columns=["Binary_Label","Multiclass_Label"])

# ---- Load model, scaler, artifacts ----
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(ART_PATH) as f:
    art = json.load(f)
selected_features = art["selected_features"]
feature_medians   = pd.Series(art["feature_medians"])
threshold         = float(art["threshold"])
target_feature    = art["target_feature"]

# ---- Build malicious set and compute anomalies with saved preprocessing ----
X_mal = X_all[y_bin == 1].copy()
y_mal_multi = y_mult[y_bin == 1].reset_index(drop=True)

y_mal_target = X_mal[target_feature].to_numpy()
X_mal_nt = X_mal.drop(columns=[target_feature])

# align features and impute with training medians
X_mal_nt = X_mal_nt.reindex(columns=selected_features, fill_value=np.nan)
X_mal_nt = X_mal_nt.fillna(feature_medians)

X_mal_scaled = scaler.transform(X_mal_nt)
y_pred = model.predict(X_mal_scaled)
errors = np.abs(y_pred - y_mal_target)
is_anom = errors > threshold

# true positives within malicious set
X_tp = X_mal_nt[is_anom].reset_index(drop=True)      # only features, no labels
y_tp = y_mal_multi[is_anom].reset_index(drop=True)   # multiclass labels for selection only
scores = errors[is_anom]                              # anomaly scores (unused in saving)

# ---- Attack classes ----
attack_mapping = {
    18: 'RRC replay attack',                               # DoS
    20: 'Bidding down with ServiceReject',                 # BiddingDown
    10: 'Location tracking via measurement reports',       # LocationTracking
    1:  'Energy Depletion attack'                          # BatteryDrain
}
attack_selection = list(attack_mapping.keys())
assert len(attack_selection) == 4, "Expected exactly 4 attack classes."

# ---- Plan per-class quota ----
per_class_target = TOTAL_TPS // len(attack_selection)  # 250 if TOTAL_TPS=1000

# ---- Collect and save per class ----
total_saved = 0
for cid in attack_selection:
    label_name = attack_mapping[cid].replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, f"{label_name}.csv")

    # All TP indices for this class
    idxs = y_tp[y_tp == cid].index.tolist()

    if not idxs:
        print(f"[!] No TP available for class {cid} ({attack_mapping[cid]}). Skipping.")
        # write empty file to keep contract consistent (optional)
        pd.DataFrame(columns=X_tp.columns).to_csv(out_path, index=False, header=False)
        continue

    # Sample without replacement up to per_class_target
    k = min(per_class_target, len(idxs))
    if k < per_class_target:
        print(f"[!] Not enough TPs for class {cid} ({attack_mapping[cid]}). "
              f"Requested {per_class_target}, available {len(idxs)}. Saving {k}.")

    chosen = random.sample(idxs, k)
    df_out = X_tp.iloc[chosen].reset_index(drop=True)

    # Save features only, no header, no index, no labels
    df_out.to_csv(out_path, index=False, header=False)
    total_saved += len(df_out)

print(f"[i] Target total: {TOTAL_TPS}, actually saved: {total_saved}")
if total_saved < TOTAL_TPS:
    print("[!] Overall total is less than requested because some classes lacked sufficient TP samples.")

# TP selection with artifacts, balanced per class

import os, json, random
import numpy as np
import pandas as pd
import joblib
from vote import Ensemble

# ---- Config ----
MODEL_PATH   = "models/new_xgb.pkl"      # model
SCALER_PATH  = "models/new_scaler.pkl"       # matching scaler
ART_PATH     = "models/new_artifacts.json"       # selected_features, medians, threshold, target
OUTPUT_DIR   = "TPs_new"
N_PER_CLASS  = 10     # number to export per class 
SEED         = 0
random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load data ----
df = pd.read_csv("data/malicious.csv", low_memory=False)
assert {"Binary_Label","Multiclass_Label"}.issubset(df.columns)

y_bin  = df["Binary_Label"].astype(int)
y_mult = df["Multiclass_Label"].astype(int)
X_all  = df.drop(columns=["Binary_Label","Multiclass_Label"])

# ---- Load model, scaler, artifacts ----
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(ART_PATH) as f:
    art = json.load(f)
selected_features = art["feature_order"]
feature_medians   = pd.Series(art["feature_medians"])
threshold         = float(art["benign_percentile_threshold"])
target_feature    = art["target_feature"]

# Save VoTE JSON for explainability 
vote_model = Ensemble.from_xgboost(model)
with open("models/new_xgb.json", "w") as f:
    f.write(vote_model.serialize())

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
X_tp = X_mal_nt[is_anom].reset_index(drop=True)
Z_tp = X_mal_scaled[is_anom]                    
y_tp = y_mal_multi[is_anom].reset_index(drop=True)
scores = errors[is_anom]                        # anomaly scores

# ---- Choose attack classes to sample from (one per macro) ----
attack_mapping = {
    18: 'RRC replay attack',                    # DoS
    20: 'Bidding down with ServiceReject',      # BiddingDown
    10: 'Location tracking via measurement reports',  # LocationTracking
    1:  'Energy Depletion attack'               # BatteryDrain
}
attack_selection = list(attack_mapping.keys())

# ---- Export per class: N_PER_CLASS , plus manifest ----
manifest = []
for cid in attack_selection:
    idxs = y_tp[y_tp == cid].index.tolist()
    if not idxs:
        print(f"[!] No TP for class {cid}")
        continue

    k = min(N_PER_CLASS, len(idxs))  
    chosen = random.sample(idxs, k)

    label_name = attack_mapping[cid].replace(" ", "_")
    for i, ridx in enumerate(chosen, 1):
        row = X_tp.iloc[[ridx]]
        suffix = f"{i}"
        fn = f"{label_name}_{suffix}.csv"
        row.to_csv(os.path.join(OUTPUT_DIR, fn), index=False, header=False)
        manifest.append({
            "filename": fn,
            "class_id": int(cid),
            "class_name": attack_mapping[cid],
            "anomaly_score": float(scores[ridx]),
            "row_index_in_tp": int(ridx)
        })

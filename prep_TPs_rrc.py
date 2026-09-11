import os, json, random
import numpy as np
import pandas as pd
import joblib

MODEL_PATH   = "models/new_xgb_rrc.pkl"
SCALER_PATH  = "models/new_scaler_rrc.pkl"
ART_PATH     = "models/new_artifacts_rrc.json"
OUTPUT_DIR   = "TPs_rrc"
N_PER_CLASS  = 10
SEED         = 0

random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load data ----
df = pd.read_csv("data/malicious_rrc.csv", low_memory=False)

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

# ---- Build malicious set and compute anomalies ----
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

# true positives
X_tp = X_mal_nt[is_anom].reset_index(drop=True)
y_tp = y_mal_multi[is_anom].reset_index(drop=True)
scores = errors[is_anom]

attack_mapping = {
    18: 'RRC replay attack',
    4: 'Paging channel hijacking attack',
    13: 'Mobile Network Mapping (MNmap)',
    1:  'Energy Depletion attack'
}
attack_selection = list(attack_mapping.keys())

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
        # enforce exact schema before writing headerless row
        row = X_tp.iloc[[ridx]].reindex(columns=selected_features)
        assert row.shape[1] == len(selected_features)

        fn = f"{label_name}_{i}.csv"
        row.to_csv(os.path.join(OUTPUT_DIR, fn), index=False, header=False)

        manifest.append({
            "filename": fn,
            "class_id": int(cid),
            "class_name": attack_mapping[cid],
            "anomaly_score": float(scores[ridx]),
            "row_index_in_tp": int(ridx)
        })

pd.DataFrame(manifest).to_csv(os.path.join(OUTPUT_DIR, "manifest.csv"), index=False)
print(f"Exported {len(manifest)} TPs to {OUTPUT_DIR}")



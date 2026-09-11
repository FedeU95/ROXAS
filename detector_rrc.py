import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# =====================
# Config
# =====================
BENIGN_PATH = "data/benign_rrc.csv"
MALICIOUS_PATH    = "data/malicious_rrc.csv"
MODEL_DIR      = "models"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

target_feature = "lte-rrc_ulInformationTransfer_element_size"


initial_benign_percentile = 85
corr_thresh = 0.95
std_thresh = 1e-6
mi_quantile = 0.25
test_size = 0.30
random_state = 0

attack_mapping = {
    0: 'Benign',
    1: 'Energy Depletion attack',
    2: 'NAS counter Desynch attack',
    3: 'X2 signalling flood',
    4: 'Paging channel hijacking attack',
    5: 'Bidding down with AttachReject',
    6: 'Incarceration with rrcReject and rrcRelease',
    7: 'Panic Attack',
    8: 'Stealthy Kickoff Attack',
    9: 'Authentication relay attack',
    10: 'Location tracking via measurement reports',
    11: 'Capability Hijacking',
    12: 'Lullaby attack using rrcReestablishRequest',
    13: 'Mobile Network Mapping (MNmap)',
    14: 'Lullaby attack with rrcResume',
    15: 'IMSI catching',
    16: 'Incarceration with rrcReestablishReject',
    17: 'Handover hijacking',
    18: 'RRC replay attack',
    19: 'Lullaby attack with rrcReconfiguration',
    20: 'Bidding down with ServiceReject',
    21: 'Bidding down with TAUReject'
}

# ========= Feature selection on benign-train only =========
def select_features(X_tr, y_tr, corr_thresh=0.95, std_thresh=1e-6, mi_quantile=0.25):
    # Only numeric for MI + correlation (safe, explicit)
    Xn = X_tr.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    med = Xn.median(numeric_only=True)
    Xn = Xn.fillna(med)

    low_var = set(Xn.columns[Xn.std(ddof=0) < std_thresh])

    mi = mutual_info_regression(Xn, y_tr, random_state=0)
    mi_s = pd.Series(mi, index=Xn.columns)

    corr = Xn.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = [(i, j, upper.loc[i, j]) for i in upper.index for j in upper.columns
             if pd.notna(upper.loc[i, j]) and upper.loc[i, j] > corr_thresh]
    pairs.sort(key=lambda x: x[2], reverse=True)

    kept = set(Xn.columns)
    to_drop = set()
    for i, j, _ in pairs:
        if i in kept and j in kept:
            drop = i if mi_s[i] < mi_s[j] else j
            kept.remove(drop)
            to_drop.add(drop)

    # Keep top (1 - mi_quantile) fraction
    mi_keep = set(mi_s[mi_s >= mi_s.quantile(1 - mi_quantile)].index)

    selected = [c for c in X_tr.columns if c not in low_var and c not in to_drop and c in mi_keep]
    if len(selected) == 0:
        raise ValueError("Feature selection removed everything. Relax thresholds.")
    return selected, mi_s.sort_values(ascending=False)

def to_numeric_and_fill(df, medians):
    Xn = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    return Xn.fillna(medians)

# ==== Load ====
benign = pd.read_csv(BENIGN_PATH, low_memory=False)
mal    = pd.read_csv(MALICIOUS_PATH, low_memory=False)

# Drop accidental index columns
for df in (benign, mal):
    if df.columns[0].startswith("Unnamed:") or df.columns[0] == "index":
        df.drop(columns=[df.columns[0]], inplace=True)

# Checks
if target_feature not in benign.columns:
    raise ValueError(f"Target '{target_feature}' not found in benign data.")
if target_feature not in mal.columns:
    raise ValueError(f"Target '{target_feature}' not found in malicious data.")
if "Multiclass_Label" not in mal.columns:
    raise ValueError("Required column 'Multiclass_Label' not found in malicious data.")

label_cols = [c for c in ["Binary_Label", "Multiclass_Label"] if c in mal.columns]

# Align malicious columns to benign + labels
benign_cols = benign.columns.tolist()
mal_aligned = mal.reindex(columns=benign_cols + label_cols)

# ==== Build benign X/y ====
y_ben = pd.to_numeric(benign[target_feature], errors="coerce")
X_ben = benign.drop(columns=[target_feature])

# Drop rows with missing target (both splits should be valid)
mask_y = y_ben.notna()
X_ben = X_ben.loc[mask_y].reset_index(drop=True)
y_ben = y_ben.loc[mask_y].reset_index(drop=True)

# ==== Split benign into train/val (NO training data used in metrics) ====
X_ben_tr, X_ben_val, y_ben_tr, y_ben_val = train_test_split(
    X_ben, y_ben, test_size=test_size, random_state=random_state
)

# ==== Malicious X/y (test) ====
y_mal = pd.to_numeric(mal_aligned[target_feature], errors="coerce")
X_mal = mal_aligned.drop(columns=[target_feature] + label_cols, errors="ignore")

# Optionally drop missing malicious target (for error computation)
mask_mal_y = y_mal.notna()
X_mal = X_mal.loc[mask_mal_y].reset_index(drop=True)
y_mal = y_mal.loc[mask_mal_y].reset_index(drop=True)

# ========= Feature selection (fit on benign train only) =========
selected_features, mi_scores = select_features(
    X_ben_tr, y_ben_tr,
    corr_thresh=corr_thresh, std_thresh=std_thresh, mi_quantile=mi_quantile
)
print(f"Features before: {X_ben_tr.shape[1]}  after: {len(selected_features)}")

# Apply selection to all sets
X_ben_tr = X_ben_tr[selected_features]
X_ben_val = X_ben_val[selected_features]
X_mal = X_mal[selected_features]

# (Optional) monotonic constraints placeholder (not used unless you pass it into XGBRegressor)
monotonicity = [1 if any(k in col for k in ['size', 'length', 'value']) else 0 for col in selected_features]

# ========= Scaling (fit on benign train only) =========
benign_train_medians = X_ben_tr.apply(lambda s: pd.to_numeric(s, errors="coerce")).median(numeric_only=True)

X_ben_tr_num = to_numeric_and_fill(X_ben_tr, benign_train_medians)
X_ben_val_num = to_numeric_and_fill(X_ben_val, benign_train_medians)
X_mal_num = to_numeric_and_fill(X_mal, benign_train_medians)

scaler = StandardScaler()
X_ben_tr_scaled = scaler.fit_transform(X_ben_tr_num)
X_ben_val_scaled = scaler.transform(X_ben_val_num)
X_mal_scaled = scaler.transform(X_mal_num)

# ========= Train regressor (benign train only) =========
model = XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=random_state,
    verbosity=0
)
model.fit(X_ben_tr_scaled, y_ben_tr)

# ========= Error-as-anomaly evaluation (VAL benign vs MAL malicious) =========
pred_ben_val = model.predict(X_ben_val_scaled)
err_ben_val = np.abs(pred_ben_val - y_ben_val.values)

pred_mal = model.predict(X_mal_scaled)
err_mal = np.abs(pred_mal - y_mal.values)

scores = np.concatenate([err_ben_val, err_mal])
y_true = np.concatenate([
    np.zeros_like(err_ben_val, dtype=int),   # benign (val)
    np.ones_like(err_mal, dtype=int)         # malicious
])

roc = roc_auc_score(y_true, scores)
pr  = average_precision_score(y_true, scores)
print(f"ROC-AUC (errors, benign-val vs malicious): {roc:.4f}")
print(f"PR-AUC  (errors, benign-val vs malicious): {pr:.4f}")

# ========= Threshold from TRAIN benign only (deployment-faithful) =========
pred_ben_tr = model.predict(X_ben_tr_scaled)
err_ben_tr = np.abs(pred_ben_tr - y_ben_tr.values)

threshold = float(np.percentile(err_ben_tr, initial_benign_percentile))
print(f"\nThreshold @ benign-train {initial_benign_percentile}th percentile: {threshold:.6g}")

# ========= Confusion-matrix-based metrics (VAL benign vs MAL malicious) =========
y_pred_class = (scores > threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
specificity = tn / (tn + fp) if (tn + fp) else 0.0
fpr = fp / (fp + tn) if (fp + tn) else 0.0

print("\n=== Global Metrics @ selected threshold (val benign vs malicious) ===")
print(f"TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
print(f"Specificity: {specificity:.4f}  FPR: {fpr:.4f}")

# ========= Per-attack detection report (malicious only) =========
y_malicious_multi = mal_aligned.loc[mask_mal_y, "Multiclass_Label"].reset_index(drop=True)
is_anomaly_mal = err_mal > threshold

print("\n=== Per-Attack Detection @ benign-train-percentile threshold ===")
for attack_id in sorted(np.unique(y_malicious_multi.dropna().astype(int))):
    name = attack_mapping.get(int(attack_id), f"Attack {int(attack_id)}")
    mask = (y_malicious_multi == attack_id)
    tot = int(mask.sum())
    det = int(is_anomaly_mal[mask].sum())
    rec = det / tot if tot else 0.0
    print(f"{name:<45} | Total: {tot:<5} | Detected: {det:<5} | Recall: {rec:.2f}")

# ========= Save artifacts =========
joblib.dump(scaler, f"{MODEL_DIR}/new_scaler_rrc.pkl")
joblib.dump(model,  f"{MODEL_DIR}/new_xgb_rrc.pkl")
artifacts = {
    "selected_features": selected_features,
    "target_feature": target_feature,
    "feature_medians": benign_train_medians.to_dict(),
    "feature_order": selected_features,  # explicit order
    "benign_percentile_threshold": threshold,
    "benign_split": {"test_size": test_size, "random_state": random_state},
    "feature_selection": {"corr_thresh": corr_thresh, "std_thresh": std_thresh, "mi_quantile": mi_quantile},
}
with open(f"{MODEL_DIR}/new_artifacts_rrc.json", "w") as f:
    json.dump(artifacts, f, indent=2)

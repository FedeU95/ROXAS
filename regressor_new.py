# Full pipeline: feature selection + benign augmentation + training + evaluation 

import os, json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import joblib

# --------------- Config ---------------
DATA_PATH = 'data/MSA.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

target_feature = 'nas-eps_emm_cause_unmaskedvalue'   #target
corr_thresh = 0.95
std_thresh = 1e-6
mi_quantile = 0.25
rng = np.random.default_rng(0)
n_aug = 3000
noise_scale = 0.01  
initial_benign_percentile = 75  # for first threshold
target_fprs = [0.01, 0.02, 0.05]  # thresholds to evaluate

# --------------- Load ---------------
data = pd.read_csv(DATA_PATH, low_memory=False)
data.drop(data.columns[0], axis=1, inplace=True)
data = data.drop_duplicates()

assert 'Binary_Label' in data.columns and 'Multiclass_Label' in data.columns
assert target_feature in data.columns, "Target feature not found in dataset!"

y_binary = data['Binary_Label']
y_multiclass = data['Multiclass_Label']
X = data.drop(columns=['Binary_Label', 'Multiclass_Label'])

X_benign = X[y_binary == 0]
X_mal   = X[y_binary == 1]
y_malicious_multi = y_multiclass[y_binary == 1]

y_train = X_benign[target_feature]
X_train = X_benign.drop(columns=[target_feature])
y_test  = X_mal[target_feature]
X_test  = X_mal.drop(columns=[target_feature])

# --------------- Data pre-processing ---------------
def select_features(X_tr, y_tr, corr_thresh=0.95, std_thresh=1e-6, mi_quantile=0.25):
    Xn = X_tr.copy().fillna(X_tr.median(numeric_only=True))
    low_var = set(Xn.columns[Xn.std(ddof=0) < std_thresh])
    mi = mutual_info_regression(Xn, y_tr, random_state=0)
    mi_s = pd.Series(mi, index=Xn.columns)

    corr = Xn.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = [(i, j, upper.loc[i, j]) for i in upper.index for j in upper.columns
             if pd.notna(upper.loc[i, j]) and upper.loc[i, j] > corr_thresh]
    pairs.sort(key=lambda x: x[2], reverse=True)
    kept = set(Xn.columns); to_drop = set()
    for i, j, _ in pairs:
        if i in kept and j in kept:
            drop = i if mi_s[i] < mi_s[j] else j
            kept.remove(drop); to_drop.add(drop)

    mi_keep = set(mi_s[mi_s >= mi_s.quantile(1 - mi_quantile)].index)
    selected = [c for c in X_tr.columns if c not in low_var and c not in to_drop and c in mi_keep]
    assert len(selected) > 0, "Feature selection removed everything. Relax thresholds."
    return selected, mi_s.sort_values(ascending=False)

selected_features, mi_scores = select_features(
    X_train, y_train, corr_thresh=corr_thresh, std_thresh=std_thresh, mi_quantile=mi_quantile
)
print(f"Features before: {X_train.shape[1]}  after: {len(selected_features)}")

with open(f'{MODEL_DIR}/AD_MSA_selected_features.json', 'w') as f:
    json.dump(selected_features, f, indent=2)
mi_scores.to_csv(f'{MODEL_DIR}/AD_MSA_mi_scores.csv')

X_train = X_train[selected_features]
X_test  = X_test[selected_features]

# Monotonic constraints for selected features
monotonicity = [1 if any(k in col for k in ['size', 'length', 'value']) else 0 for col in selected_features]

# --------------- Benign augmentation ---------------
X_train_fs = X_train.fillna(X_train.median(numeric_only=True))
feat_std = X_train_fs.std(ddof=0).replace(0, 1e-12).values
boot_idx = rng.integers(low=0, high=len(X_train_fs), size=n_aug)
X_boot = X_train_fs.iloc[boot_idx].to_numpy()
y_boot = np.asarray(y_train.iloc[boot_idx])
noise = rng.normal(0.0, noise_scale, size=X_boot.shape) * feat_std
X_aug = X_boot + noise
y_aug = y_boot

X_train_aug = pd.DataFrame(np.vstack([X_train_fs.to_numpy(), X_aug]), columns=X_train.columns)
y_train_aug = np.concatenate([y_train.to_numpy(), y_aug])
print(f"Augmented benign samples: +{n_aug}  → total benign train: {len(y_train_aug)}")

# --------------- Scale ---------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_aug)
X_test_scaled  = scaler.transform(X_test.fillna(X_train.median(numeric_only=True)))
joblib.dump(scaler, f'{MODEL_DIR}/AD_MSA_scaler_aug.pkl')


# --------------- Train ---------------
model = XGBRegressor(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    monotone_constraints=tuple(monotonicity),
    verbosity=0,
    random_state=0
)
model.fit(X_train_scaled, y_train_aug)

model_path = f'{MODEL_DIR}/AD_MSA_nas_XGB_aug.pkl'
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

# --------------- Scores ---------------
val_pred = model.predict(X_train_scaled)
val_errors = np.abs(val_pred - y_train_aug)          # benign errors
y_pred = model.predict(X_test_scaled)
errors = np.abs(y_pred - y_test)                      # malicious errors

threshold = np.percentile(val_errors, initial_benign_percentile)

def global_metrics(val_err, mal_err, th):
    y_true = np.concatenate([np.zeros_like(val_err, dtype=int), np.ones_like(mal_err, dtype=int)])
    y_scores = np.concatenate([val_err, mal_err])
    y_pred = (y_scores > th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    spec      = tn / (tn + fp) if (tn + fp) else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) else 0.0
    roc_auc   = roc_auc_score(y_true, y_scores)
    pr_auc    = average_precision_score(y_true, y_scores)
    return dict(threshold=th, tp=tp, fp=fp, tn=tn, fn=fn,
                precision=precision, recall=recall, f1=f1,
                specificity=spec, fpr=fpr, roc_auc=roc_auc, pr_auc=pr_auc)

def th_from_target_fpr(val_err, target_fpr):
    # choose threshold so that fraction of benign > th equals target_fpr
    val_sorted = np.sort(val_err)
    idx = int(np.ceil((1 - target_fpr) * len(val_sorted))) - 1
    idx = np.clip(idx, 0, len(val_sorted) - 1)
    return float(val_sorted[idx])

# --------------- Report ---------------
gm = global_metrics(val_errors, errors, threshold)
print(f"\n=== Global metrics @ threshold {gm['threshold']:.6g} ===")
print(f"TP: {gm['tp']}  FP: {gm['fp']}  TN: {gm['tn']}  FN: {gm['fn']}")
print(f"Precision: {gm['precision']:.4f}  Recall: {gm['recall']:.4f}  F1: {gm['f1']:.4f}")
print(f"Specificity: {gm['specificity']:.4f}  FPR: {gm['fpr']:.4f}")
print(f"ROC-AUC: {gm['roc_auc']:.4f}  PR-AUC: {gm['pr_auc']:.4f}")

# --------------- Thresholds at target FPRs ---------------
print("\n=== Thresholds by target benign FPR ===")
best_by_f1 = None
for tfpr in target_fprs:
    th = th_from_target_fpr(val_errors, tfpr)
    m = global_metrics(val_errors, errors, th)
    if best_by_f1 is None or m['f1'] > best_by_f1['f1']:
        best_by_f1 = m
    print(f"Target FPR {tfpr:>5.2%} → th={m['threshold']:.6g}  "
          f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  F1={m['f1']:.4f}  "
          f"Spec={m['specificity']:.4f}  FPR={m['fpr']:.4f}")

print("\n=== Best threshold by F1 among target FPRs ===")
print(f"th={best_by_f1['threshold']:.6g}  Prec={best_by_f1['precision']:.4f}  Rec={best_by_f1['recall']:.4f}  "
      f"F1={best_by_f1['f1']:.4f}  Spec={best_by_f1['specificity']:.4f}  FPR={best_by_f1['fpr']:.4f}")

# --------------- Per-attack recall at chosen threshold ---------------
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

print("\n=== Per-Attack Detection Report @ best F1 threshold ===")
is_anomaly_best = errors > best_by_f1['threshold']
for attack_id in sorted(np.unique(y_malicious_multi)):
    name = attack_mapping.get(int(attack_id), f"Attack {int(attack_id)}")
    mask = (y_malicious_multi == attack_id)
    tot = int(np.sum(mask))
    det = int(np.sum(is_anomaly_best[mask]))
    rec = det / tot if tot else 0.0
    print(f"{name:<45} | Total: {tot:<5} | Detected: {det:<5} | Recall: {rec:.2f}")

feature_medians = X_train_fs.median(numeric_only=True).to_dict()

# Collect all artifacts
artifacts = {
    "selected_features": selected_features,
    "feature_medians": feature_medians,
    "threshold": float(threshold),
    "target_feature": target_feature
}

with open("models/AD_MSA_artifacts.json", "w") as f:
    json.dump(artifacts, f, indent=2)

print("Artifacts saved to models/AD_MSA_artifacts.json")
import os, json, glob
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from itertools import combinations
from vote import Ensemble

# ---------- Config ----------
MODEL_PATH   = "models/AD_MSA_nas_XGB_aug.pkl"
SCALER_PATH  = "models/AD_MSA_scaler_aug.pkl"
ART_PATH     = "models/AD_MSA_artifacts.json"

TP_DIR       = "TPs_eval"   
TOP_K        = None         

OUT_DIR      = "eval_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load artifacts ----------
with open(ART_PATH) as f:
    art = json.load(f)
selected_features = art["selected_features"]
feature_medians   = pd.Series(art["feature_medians"])
target_feature    = art["target_feature"]

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
vote_model = Ensemble.from_xgboost(model)

# ---------- Helpers ----------
def read_tp_file(path, cols):
    """
    Read a TP CSV saved without header/labels and assign the training-time
    selected feature order.
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(cols):
        raise ValueError(f"Column count mismatch in {path}: got {df.shape[1]}, expected {len(cols)}")
    df.columns = cols
    return df

def class_name_from_filename(fn):
    base = os.path.splitext(os.path.basename(fn))[0]
    return base

# ---- metrics helpers (sparsity & stability) ----
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def compute_sparsity_stability(rows):
    """
    rows: list[{"class_name": str, "row_idx": int, "feature_set": set[str]}]
    Returns (per_class_df, overall_summary_dict).
    """
    df = pd.DataFrame(rows)
    metrics = []

    for cls, grp in df.groupby("class_name"):
        sets = grp["feature_set"].tolist()
        sizes = [len(s) for s in sets]
        sparsity_mean = float(np.mean(sizes)) if sizes else float("nan")
        sparsity_std  = float(np.std(sizes, ddof=1)) if len(sizes) > 1 else 0.0

        pair_idxs = list(combinations(range(len(sets)), 2))
        if pair_idxs:
            sims = [jaccard(sets[i], sets[j]) for i, j in pair_idxs]
            stability_mean = float(np.mean(sims))
            stability_median = float(np.median(sims))
            stability_std = float(np.std(sims, ddof=1)) if len(sims) > 1 else 0.0
            num_pairs = len(sims)
        else:
            stability_mean = float("nan")
            stability_median = float("nan")
            stability_std = float("nan")
            num_pairs = 0

        metrics.append({
            "class_name": cls,
            "num_samples": len(sets),
            "sparsity_mean": sparsity_mean,
            "sparsity_std": sparsity_std,
            "stability_jaccard_mean": stability_mean,
            "stability_jaccard_median": stability_median,
            "stability_jaccard_std": stability_std,
            "num_pairs": num_pairs,
        })

    per_class_df = pd.DataFrame(metrics).sort_values("class_name")

    # Overall metrics
    all_sets = df["feature_set"].tolist()
    all_sizes = [len(s) for s in all_sets]
    overall_sparsity_mean = float(np.mean(all_sizes)) if all_sizes else float("nan")

    # Weighted mean of per-class stability by number of pairs
    if len(per_class_df.dropna(subset=["stability_jaccard_mean"])) > 0 and per_class_df["num_pairs"].sum() > 0:
        w = per_class_df["num_pairs"].astype(float)
        overall_stability_mean = float((per_class_df["stability_jaccard_mean"] * w).sum() / w.sum())
    else:
        overall_stability_mean = float("nan")

    overall = {
        "overall_sparsity_mean": overall_sparsity_mean,
        "overall_stability_jaccard_mean": overall_stability_mean
    }
    return per_class_df, overall

# ---------- Gather TPs ----------
tp_files = sorted(glob.glob(os.path.join(TP_DIR, "*.csv")))
if not tp_files:
    raise SystemExit(f"No TP files found in {TP_DIR}")

explain_rows = []    # per-sample feature sets for sparsity/stability

for fp in tp_files:
    cls_name = class_name_from_filename(fp)

    # Load all rows for this class
    X_cls = read_tp_file(fp, selected_features)

    # Impute any unexpected NaNs
    X_cls = X_cls.fillna(feature_medians)

    # Scale once for efficiency
    X_scaled = pd.DataFrame(
        scaler.transform(X_cls),
        columns=selected_features,
        index=X_cls.index
    )

    # Optional warm-up (avoids first-call overhead skew)
    _ = vote_model.explain_minimal(X_scaled.values[0])

    # Explain each row 
    for i in range(len(X_cls)):
        x_row_unscaled = X_cls.iloc[i]
        x_row_scaled = X_scaled.values[i]

        feat_idx = vote_model.explain_minimal(x_row_scaled)

        if TOP_K is not None:
            feat_idx = feat_idx[:TOP_K]

        # Collect metrics info
        feature_set = {selected_features[j] for j in feat_idx}

        explain_rows.append({
            "class_name": cls_name,
            "row_idx": i,
            "feature_set": feature_set
        })


# ---------- Compute & save sparsity/stability ----------
per_class_metrics, overall_metrics = compute_sparsity_stability(explain_rows)
metrics_csv = os.path.join(OUT_DIR, "explanation_metrics.csv")
per_class_metrics.to_csv(metrics_csv, index=False)
print("[i] Explanation metrics (per class):")
print(per_class_metrics.to_string(index=False))
print("[i] Overall metrics:")
print(overall_metrics)
print(f"[i] Saved explanation metrics to {metrics_csv}")


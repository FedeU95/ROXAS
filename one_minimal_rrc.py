import os, json, glob
import numpy as np
import pandas as pd
import joblib
from vote import Ensemble

# ---------- Config ----------
MODEL_PATH  = "models/new_xgb_rrc.pkl"
SCALER_PATH    = "models/new_scaler_rrc.pkl"
ART_PATH       = "models/new_artifacts_rrc.json"
TP_DIR         = "TPs_rrc"
OUT_DIR        = "prompts_rrc"
TOP_K          = None   # set e.g. 10 to cap number of flagged features; None = all returned
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load artifacts ----------
with open(ART_PATH) as f:
    art = json.load(f)
selected_features = art["feature_order"]
feature_medians   = pd.Series(art["feature_medians"])
threshold         = float(art["benign_percentile_threshold"])
target_feature    = art["target_feature"]

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
vote_model = Ensemble.from_xgboost(model)


# ---------- Helper ----------
def load_tp_csv(path, cols):
    # Files were saved without header; enforce selected_features order
    row = pd.read_csv(path, header=None, nrows=1)
    assert row.shape[1] == len(cols), f"Column count mismatch in {path}"
    row.columns = cols
    return row

def write_prompt_txt(path, pairs):
    with open(path, "w") as f:
        f.write("The following feature values were flagged:\n\n")
        for name, val in pairs:
            # keep numeric formatting stable
            if isinstance(val, float) and (val.is_integer()):
                val = int(val)
            f.write(f"- {name}={val}\n")

manifest_rows = []

# ---------- Iterate TPs ----------
tp_files = sorted(glob.glob(os.path.join(TP_DIR, "*.csv")))
if not tp_files:
    raise SystemExit(f"No TP files found in {TP_DIR}")

for fp in tp_files:
    # load unscaled row
    x_row = load_tp_csv(fp, selected_features)

    # impute in case any NaNs slipped in
    x_row = x_row.fillna(feature_medians)

    # scale and explain
    x_scaled = pd.DataFrame(scaler.transform(x_row), columns=selected_features, index=x_row.index)

    # VoTE minimal explanation returns indices of features to change
    feat_idx = vote_model.explain_minimal(x_scaled.values[0])
    if TOP_K is not None:
        feat_idx = feat_idx[:TOP_K]

    flagged = [(selected_features[i], x_row.iloc[0, i]) for i in feat_idx]

    # write text file next to name
    base = os.path.splitext(os.path.basename(fp))[0]
    out_txt = os.path.join(OUT_DIR, f"{base}.txt")
    write_prompt_txt(out_txt, flagged)

    manifest_rows.append({
        "tp_file": os.path.basename(fp),
        "prompt_file": os.path.basename(out_txt),
        "num_flagged": len(flagged),
        "features": ", ".join([n for n, _ in flagged])
    })

# ---------- Save manifest ----------
pd.DataFrame(manifest_rows).to_csv(os.path.join(OUT_DIR, "manifest_explanations.csv"), index=False)
print(f"Wrote {len(manifest_rows)} prompt files to {OUT_DIR}")

# TP-class selection: 1 per macrocategory with representation + diversity

import os, json, random
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_distances

# ---------------- Config ----------------
DATA_PATH   = "data/MSA.csv"
MODEL_PATH  = "models/AD_MSA_nas_XGB_aug.pkl"
SCALER_PATH = "models/AD_MSA_scaler_aug.pkl"
ART_PATH    = "models/AD_MSA_artifacts.json"

MIN_TP = 10            # require >= MIN_TP true positives per class
RANDOM_SEED = 0
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

attack_mapping = {
    0:'Benign',1:'Energy Depletion attack',2:'NAS counter Desynch attack',3:'X2 signalling flood',
    4:'Paging channel hijacking attack',5:'Bidding down with AttachReject',6:'Incarceration with rrcReject and rrcRelease',
    7:'Panic Attack',8:'Stealthy Kickoff Attack',9:'Authentication relay attack',10:'Location tracking via measurement reports',
    11:'Capability Hijacking',12:'Lullaby attack using rrcReestablishRequest',13:'Mobile Network Mapping (MNmap)',
    14:'Lullaby attack with rrcResume',15:'IMSI catching',16:'Incarceration with rrcReestablishReject',
    17:'Handover hijacking',18:'RRC replay attack',19:'Lullaby attack with rrcReconfiguration',
    20:'Bidding down with ServiceReject',21:'Bidding down with TAUReject'
}

# --- Macro categories ---
macro_mapping = {
    # DoS
    2:"DoS",9:"DoS",17:"DoS",18:"DoS",16:"DoS",
    # Bidding Down
    5:"BiddingDown",21:"BiddingDown",20:"BiddingDown",
    # Location Tracking
    10:"LocationTracking",
    # Battery Drain
    1:"BatteryDrain",3:"BatteryDrain",4:"BatteryDrain",6:"BatteryDrain",7:"BatteryDrain",
    8:"BatteryDrain",11:"BatteryDrain",12:"BatteryDrain",13:"BatteryDrain",14:"BatteryDrain",
    15:"BatteryDrain",19:"BatteryDrain"
}
macro_order = ["DoS","BiddingDown","LocationTracking","BatteryDrain"]  

# ---------------- Load artifacts ----------------
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(ART_PATH) as f:
    art = json.load(f)
selected_features = art["selected_features"]
feature_medians   = pd.Series(art["feature_medians"])
threshold         = float(art["threshold"])
target_feature    = art["target_feature"]

# ---------------- Load data ----------------
df = pd.read_csv(DATA_PATH, low_memory=False)
df.drop(df.columns[0], axis=1, inplace=True)
df = df.drop_duplicates()
assert {'Binary_Label','Multiclass_Label',target_feature}.issubset(df.columns)

y_bin  = df['Binary_Label'].astype(int)
y_mul  = df['Multiclass_Label'].astype(int)
X_all  = df.drop(columns=['Binary_Label','Multiclass_Label'])

# ---------------- Build malicious TP set ----------------
X_mal = X_all[y_bin == 1].copy()
y_mal_multi = y_mul[y_bin == 1].reset_index(drop=True)

y_mal_target = X_mal[target_feature].to_numpy()
X_mal_nt = X_mal.drop(columns=[target_feature]).reindex(columns=selected_features, fill_value=np.nan)
X_mal_nt = X_mal_nt.fillna(feature_medians)

X_mal_scaled = scaler.transform(X_mal_nt)
y_pred = model.predict(X_mal_scaled)
errors = np.abs(y_pred - y_mal_target)
is_anom = errors > threshold

# TPs
X_tp_scaled = X_mal_scaled[is_anom]
y_tp_multi  = y_mal_multi[is_anom].reset_index(drop=True)

# ---------------- Representation ----------------
tp_counts = y_tp_multi.value_counts().sort_values(ascending=False)
valid_classes = [c for c in tp_counts.index if tp_counts[c] >= MIN_TP and c in macro_mapping]

print("TP counts per class (desc):")
print(tp_counts)
print(f"\nValid classes (>= {MIN_TP} TPs): {valid_classes}")

if not valid_classes:
    raise SystemExit("No classes meet MIN_TP. Lower MIN_TP or improve detector.")

# ---------------- Centroids per valid class ----------------
centroids = {}
for cid in valid_classes:
    mask = (y_tp_multi == cid).to_numpy()
    centroids[cid] = X_tp_scaled[mask].mean(axis=0)

# ---------------- Select 1 class per macro ----------------
chosen = []
chosen_by_macro = {}

def min_dist_to_chosen(vec, chosen_centroids):
    if not chosen_centroids:
        return np.inf
    d = cosine_distances(vec.reshape(1,-1), np.vstack(chosen_centroids)).flatten()
    return float(d.min())

chosen_centroids = []
for macro in macro_order:
    cand = [c for c in valid_classes if macro_mapping.get(int(c)) == macro]
    if not cand:
        print(f"[!] No valid classes for macro {macro}")
        continue
    # score = max-min distance to already chosen; tie-break by TP count
    best_c, best_score, best_tp = None, -1.0, -1
    for c in cand:
        score = min_dist_to_chosen(centroids[c], chosen_centroids)
        tp = int(tp_counts[c])
        if (score > best_score) or (np.isclose(score, best_score) and tp > best_tp):
            best_c, best_score, best_tp = c, score, tp
    chosen.append(best_c)
    chosen_by_macro[macro] = int(best_c)
    chosen_centroids.append(centroids[best_c])

# ---------------- Report ----------------
print("\nChosen classes (one per macro):")
for macro in macro_order:
    cid = chosen_by_macro.get(macro)
    if cid is None: 
        continue
    print(f"- {macro}: {cid} | {attack_mapping.get(int(cid), str(cid))} | TP count: {int(tp_counts[cid])}")

# Save selection
out = {"chosen_by_macro": chosen_by_macro, "min_tp": MIN_TP}
with open("models/chosen_classes_by_macro.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved: models/chosen_classes_by_macro.json")



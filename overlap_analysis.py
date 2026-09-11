import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# === Load feature names ===
csv_path = 'data/MSA.csv'
df = pd.read_csv(csv_path)
df.drop(df.columns[0], axis=1, inplace=True)
df.drop(columns=['Binary_Label', 'Multiclass_Label'], inplace=True)
feature_names = df.columns.tolist()
nb_features = len(feature_names)

# === Explanation directory ===
explanation_dir = "explanations"
explanation_files = [f for f in os.listdir(explanation_dir) if f.endswith('.json')]

# === Feature usage by attack ===
attack_feature_usage = defaultdict(lambda: np.zeros(nb_features, dtype=int))

# === Reverse map for readability
attack_mapping = {
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

# === Process each explanation file ===
for file in explanation_files:
    path = os.path.join(explanation_dir, file)
    with open(path) as f:
        obj = json.load(f)

    attack_name = next((v for k, v in attack_mapping.items() if v.replace(" ", "_").replace("-", "_").lower() in file.lower()), None)
    if not attack_name:
        continue

    usage = np.zeros(nb_features, dtype=int)
    for explanation in obj[0]['explanations']:
        for idx in explanation:
            usage[idx] += 1

    attack_feature_usage[attack_name] += usage

# === Convert to DataFrame ===
usage_df = pd.DataFrame.from_dict(attack_feature_usage, orient='index', columns=feature_names)

# === Normalize if needed (e.g., top-k only) ===
top_k = 10
top_features_per_attack = {}
for attack, row in usage_df.iterrows():
    top_indices = row.argsort()[-top_k:][::-1]
    top_features_per_attack[attack] = [feature_names[i] for i in top_indices if row[i] > 0]

# === Analyze Overlap ===
from itertools import combinations
from collections import Counter

overlap_counter = Counter()

for attack1, attack2 in combinations(top_features_per_attack.keys(), 2):
    shared = set(top_features_per_attack[attack1]) & set(top_features_per_attack[attack2])
    if shared:
        overlap_counter[frozenset((attack1, attack2))] = len(shared)

# === Show results ===
print("\n=== Top Features per Attack ===")
for attack, feats in top_features_per_attack.items():
    print(f"{attack:45}: {feats}")

print("\n=== Overlap Between Attacks (shared top-k features) ===")
for pair, count in overlap_counter.most_common():
    atk1, atk2 = tuple(pair)
    print(f"{atk1:40} <-> {atk2:40} | Shared: {count}")

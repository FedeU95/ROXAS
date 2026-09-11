import os
import glob
import pandas as pd

NAS_DIR = "prompts_new"
RRC_DIR = "prompts_rrc"


import re

def infer_attack_name(filename):
    name = os.path.splitext(os.path.basename(filename))[0]

    # remove trailing sample number
    name = re.sub(r'_\d+$', '', name)

    return name

def read_features_from_prompt(path):
    features = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("-") and "=" in line:
                feature = line[1:].split("=", 1)[0].strip()
                features.append(feature)

    return features


def analyze_prompt_dir(prompt_dir, layer):

    rows = []

    for path in sorted(glob.glob(os.path.join(prompt_dir, "*.txt"))):

        features = read_features_from_prompt(path)

        rows.append({
            "layer": layer,
            "file": os.path.basename(path),
            "attack": infer_attack_name(path),
            "num_features": len(features),
            "features": features
        })

    df = pd.DataFrame(rows)

    all_features = []
    for feats in df["features"]:
        all_features.extend(feats)

    freq = pd.Series(all_features).value_counts()

    total_occurrences = freq.sum()

    summary = {
        "Layer": layer,
        "Explanations": len(df),
        "Avg explanation size": round(df["num_features"].mean(), 2),
        "Min size": df["num_features"].min(),
        "Max size": df["num_features"].max(),
        "Unique features": len(freq),
        "Top-1 coverage (%)": round(
            100 * freq.iloc[0] / total_occurrences, 1
        ),
        "Top-2 coverage (%)": round(
            100 * freq.head(2).sum() / total_occurrences, 1
        ),
        "Top-5 coverage (%)": round(
            100 * freq.head(min(5, len(freq))).sum() / total_occurrences, 1
        )
    }

    return df, freq, summary


all_summaries = []

for folder, layer in [(NAS_DIR, "NAS"), (RRC_DIR, "RRC")]:

    df, freq, summary = analyze_prompt_dir(folder, layer)

    all_summaries.append(summary)

    print("\n" + "=" * 60)
    print(layer)
    print("=" * 60)

    print("\nExplanation size per attack:")
    print(
        df.groupby("attack")["num_features"]
        .agg(["count", "mean", "min", "max"])
    )

    print("\nTop features:")
    print(freq.head(15))

summary_df = pd.DataFrame(all_summaries)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(summary_df)

summary_df.to_csv("explanation_quality_summary.csv", index=False)
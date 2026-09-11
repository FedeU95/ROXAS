import re
import json
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# ================= CONFIG =================

HUMAN_CSV = Path("validation/normalized_all_validators.csv")
LLM_JUDGE_CSV = Path("llm_judge_unconstrained/llm_judge_long_format.csv")
CONSTRAINED_CSV = Path("llm_mitigation_exp_constrained/llm_constrained_long_format.csv")
FIGHT_MITIGATIONS_JSON = Path("fight_db/fight_mitigations.json")

OUT_DIR = Path("constrained_eval")
OUT_DIR.mkdir(exist_ok=True)

OUT_ALERT_LEVEL = OUT_DIR / "alert_level_constrained_eval.csv"
OUT_ATTACK_LEVEL = OUT_DIR / "attack_level_constrained_eval.csv"
OUT_INVALID_FGMS = OUT_DIR / "invalid_fgms.csv"
OUT_SUMMARY = OUT_DIR / "summary.csv"
OUT_FGM_FREQ = OUT_DIR / "fgm_frequency.csv"
OUT_ATTACK_SPECIFICITY = OUT_DIR / "attack_specificity_constrained.csv"


# ================= HELPERS =================

def canonical_attack(x):
    if pd.isna(x):
        return ""

    s = str(x).lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    if "bidding" in s and "service" in s:
        return "bidding_down_service_reject"
    if "location" in s:
        return "location_tracking"
    if "rrc" in s and "replay" in s:
        return "rrc_replay"
    if "energy" in s:
        return "energy_depletion"

    return s


def extract_alert_id(x):
    if pd.isna(x):
        return ""

    s = str(x)
    m = re.search(r"(\d+)$", s)

    if not m:
        m = re.search(r"(\d+)", s)

    return str(int(m.group(1))) if m else ""


def fgm_set(x):
    if pd.isna(x):
        return set()

    text = str(x).upper().strip()

    if text in {"", "NAN", "NONE", "NULL", "NO_MATCH", "UNKNOWN", "N/A", "NA"}:
        return set()

    return set(re.findall(r"FGM\d+(?:\.\d+)?", text))


def fmt(s):
    return ";".join(sorted(s))


def normalize_text(x):
    if pd.isna(x):
        return ""

    s = str(x).lower()
    s = re.sub(r"fgm\d+(?:\.\d+)?", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def text_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a and not b:
        return None
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def best_mean_similarity(source_texts, reference_texts):
    if not source_texts or not reference_texts:
        return None

    scores = []

    for src in source_texts:
        scores.append(max(text_similarity(src, ref) or 0.0 for ref in reference_texts))

    return sum(scores) / len(scores)


def text_list(df):
    if df.empty or "mitigation_text" not in df.columns:
        return []

    return [
        str(x).strip()
        for x in df["mitigation_text"].dropna().tolist()
        if str(x).strip()
    ]


def union_fgms(df):
    if df.empty or "mapped_fgm_ids" not in df.columns:
        return set()

    out = set()

    for x in df["mapped_fgm_ids"]:
        out |= fgm_set(x)

    return out


def load_valid_fgms():
    with open(FIGHT_MITIGATIONS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid = set()

    for item in data:
        for key in ["mitigation_id", "id"]:
            if item.get(key):
                valid.add(str(item[key]).upper().strip())

    return valid


def fgm_frequency(df):
    all_fgms = []

    for x in df["mapped_fgm_ids"]:
        all_fgms.extend(sorted(fgm_set(x)))

    freq = pd.Series(all_fgms).value_counts()

    if freq.empty:
        return pd.DataFrame(columns=["fgm_id", "count", "coverage"])

    total = freq.sum()

    return pd.DataFrame({
        "fgm_id": freq.index,
        "count": freq.values,
        "coverage": freq.values / total,
    })


def fgm_diversity_metrics(df):
    freq_df = fgm_frequency(df)

    if freq_df.empty:
        return {
            "n_total_fgm_occurrences": 0,
            "n_unique_fgms": 0,
            "top1_fgm_coverage": None,
            "top3_fgm_coverage": None,
            "top5_fgm_coverage": None,
            "top_fgms": "",
        }

    total = freq_df["count"].sum()

    return {
        "n_total_fgm_occurrences": int(total),
        "n_unique_fgms": int(len(freq_df)),
        "top1_fgm_coverage": freq_df["count"].head(1).sum() / total,
        "top3_fgm_coverage": freq_df["count"].head(3).sum() / total,
        "top5_fgm_coverage": freq_df["count"].head(5).sum() / total,
        "top_fgms": ";".join(
            f"{row.fgm_id}:{int(row.count)}"
            for row in freq_df.head(10).itertuples(index=False)
        ),
    }


def attack_specific_fgm_metrics(df):
    attack_sets = {
        attack: union_fgms(group)
        for attack, group in df.groupby("attack_key")
    }

    shared_with_all = (
        set.intersection(*attack_sets.values())
        if attack_sets
        else set()
    )

    rows = []

    for attack, fgms in attack_sets.items():
        other_fgms = set().union(
            *[s for a, s in attack_sets.items() if a != attack]
        )

        unique_to_attack = fgms - other_fgms

        rows.append({
            "attack_key": attack,
            "n_attack_fgms": len(fgms),
            "attack_fgms": fmt(fgms),
            "n_fgms_unique_to_attack": len(unique_to_attack),
            "fgms_unique_to_attack": fmt(unique_to_attack),
            "n_fgms_shared_with_all_attacks": len(shared_with_all),
            "fgms_shared_with_all_attacks": fmt(shared_with_all),
        })

    return pd.DataFrame(rows)


# ================= LOAD =================

human = pd.read_csv(HUMAN_CSV)
judge = pd.read_csv(LLM_JUDGE_CSV)
constrained = pd.read_csv(CONSTRAINED_CSV)

valid_fgms = load_valid_fgms()

human["attack_key"] = human["attack_key"].apply(canonical_attack)
human["alert_id"] = human["alert_id"].apply(extract_alert_id)

judge["attack_key"] = judge["sample_id"].apply(canonical_attack)
judge["alert_id"] = judge["sample_id"].apply(extract_alert_id)

constrained["attack_key"] = constrained["attack_key"].apply(canonical_attack)
constrained["alert_id"] = constrained["alert_id"].apply(extract_alert_id)


# ================= INVALID FGM CHECK =================

invalid_rows = []

for _, row in constrained.iterrows():
    ids = fgm_set(row.get("mapped_fgm_ids"))

    for fgm in ids:
        if fgm not in valid_fgms:
            invalid_rows.append({
                "attack_key": row.get("attack_key"),
                "alert_id": row.get("alert_id"),
                "sample_id": row.get("sample_id"),
                "mitigation_index": row.get("mitigation_index"),
                "invalid_fgm": fgm,
                "mapped_fgm_ids": row.get("mapped_fgm_ids"),
                "mitigation_text": row.get("mitigation_text"),
            })

invalid_df = pd.DataFrame(invalid_rows)
invalid_df.to_csv(OUT_INVALID_FGMS, index=False)


# ================= FGM FREQUENCY =================

freq_out = fgm_frequency(constrained)
freq_out.to_csv(OUT_FGM_FREQ, index=False)


# ================= ATTACK SPECIFICITY =================

attack_specificity_out = attack_specific_fgm_metrics(constrained)
attack_specificity_out.to_csv(OUT_ATTACK_SPECIFICITY, index=False)


# ================= ALERT-LEVEL EVAL =================

all_alerts = sorted(
    set(map(tuple, constrained[["attack_key", "alert_id"]].drop_duplicates().values))
)

alert_rows = []

for attack_key, alert_id in all_alerts:
    cdf = constrained[
        (constrained["attack_key"] == attack_key)
        & (constrained["alert_id"] == alert_id)
    ]

    hdf = human[
        (human["attack_key"] == attack_key)
        & (human["alert_id"] == alert_id)
    ]

    jdf = judge[
        (judge["attack_key"] == attack_key)
        & (judge["alert_id"] == alert_id)
    ]

    c_fgms = union_fgms(cdf)
    h_fgms = union_fgms(hdf)
    j_fgms = union_fgms(jdf)

    c_texts = text_list(cdf)
    unconstrained_texts = text_list(jdf)

    invalid = sorted([x for x in c_fgms if x not in valid_fgms])

    alert_rows.append({
        "attack_key": attack_key,
        "alert_id": alert_id,

        "has_human_reference": not hdf.empty,
        "has_llm_judge_reference": not jdf.empty,

        "n_constrained_actions": len(c_texts),
        "n_unconstrained_actions": len(unconstrained_texts),

        "n_constrained_fgms": len(c_fgms),
        "n_human_fgms": len(h_fgms),
        "n_llm_judge_fgms": len(j_fgms),

        "constrained_fgms": fmt(c_fgms),
        "human_fgms": fmt(h_fgms),
        "llm_judge_fgms": fmt(j_fgms),

        "all_constrained_fgms_valid": len(invalid) == 0,
        "invalid_constrained_fgms": ";".join(invalid),

        "constrained_vs_unconstrained_text_similarity": best_mean_similarity(
            c_texts,
            unconstrained_texts,
        ),

        "constrained_human_fgm_overlap": len(c_fgms & h_fgms) > 0,
        "n_constrained_human_shared_fgms": len(c_fgms & h_fgms),
        "constrained_human_shared_fgms": fmt(c_fgms & h_fgms),

        "constrained_judge_fgm_overlap": len(c_fgms & j_fgms) > 0,
        "n_constrained_judge_shared_fgms": len(c_fgms & j_fgms),
        "constrained_judge_shared_fgms": fmt(c_fgms & j_fgms),
    })

alert_out = pd.DataFrame(alert_rows)
alert_out.to_csv(OUT_ALERT_LEVEL, index=False)


# ================= ATTACK-LEVEL SUMMARY =================

attack_rows = []

for attack_key, g in alert_out.groupby("attack_key"):
    constrained_attack = constrained[constrained["attack_key"] == attack_key]
    div = fgm_diversity_metrics(constrained_attack)

    attack_rows.append({
        "attack_key": attack_key,
        "n_alerts": len(g),
        "n_human_reference_alerts": int(g["has_human_reference"].sum()),
        "n_llm_judge_reference_alerts": int(g["has_llm_judge_reference"].sum()),

        "valid_fgm_rate": g["all_constrained_fgms_valid"].mean(),

        "mean_constrained_actions_per_alert": g["n_constrained_actions"].mean(),
        "mean_constrained_fgms_per_alert": g["n_constrained_fgms"].mean(),

        "n_total_fgm_occurrences": div["n_total_fgm_occurrences"],
        "n_unique_fgms": div["n_unique_fgms"],
        "top1_fgm_coverage": div["top1_fgm_coverage"],
        "top3_fgm_coverage": div["top3_fgm_coverage"],
        "top5_fgm_coverage": div["top5_fgm_coverage"],
        "top_fgms": div["top_fgms"],

        "mean_text_similarity_to_unconstrained": g[
            "constrained_vs_unconstrained_text_similarity"
        ].mean(),

        "human_fgm_overlap_rate": g.loc[
            g["has_human_reference"],
            "constrained_human_fgm_overlap"
        ].mean(),

        "mean_shared_fgms_with_human": g.loc[
            g["has_human_reference"],
            "n_constrained_human_shared_fgms"
        ].mean(),

        "judge_fgm_overlap_rate": g.loc[
            g["has_llm_judge_reference"],
            "constrained_judge_fgm_overlap"
        ].mean(),

        "mean_shared_fgms_with_judge": g.loc[
            g["has_llm_judge_reference"],
            "n_constrained_judge_shared_fgms"
        ].mean(),
    })

attack_out = pd.DataFrame(attack_rows)
attack_out.to_csv(OUT_ATTACK_LEVEL, index=False)


# ================= OVERALL SUMMARY =================

div = fgm_diversity_metrics(constrained)

attack_sets = {
    attack: union_fgms(group)
    for attack, group in constrained.groupby("attack_key")
}

shared_all = (
    set.intersection(*attack_sets.values())
    if attack_sets
    else set()
)

unique_any = {
    attack: attack_sets[attack] - set().union(
        *[s for a, s in attack_sets.items() if a != attack]
    )
    for attack in attack_sets
}

summary = {
    "n_constrained_alerts": len(alert_out),
    "n_human_reference_alerts": int(alert_out["has_human_reference"].sum()),
    "n_llm_judge_reference_alerts": int(alert_out["has_llm_judge_reference"].sum()),

    "n_invalid_constrained_fgms": len(invalid_df),
    "valid_fgm_rate": alert_out["all_constrained_fgms_valid"].mean(),

    "mean_constrained_actions_per_alert": alert_out["n_constrained_actions"].mean(),
    "mean_constrained_fgms_per_alert": alert_out["n_constrained_fgms"].mean(),

    "n_total_fgm_occurrences": div["n_total_fgm_occurrences"],
    "n_unique_fgms": div["n_unique_fgms"],
    "top1_fgm_coverage": div["top1_fgm_coverage"],
    "top3_fgm_coverage": div["top3_fgm_coverage"],
    "top5_fgm_coverage": div["top5_fgm_coverage"],
    "top_fgms": div["top_fgms"],

    "n_fgms_shared_by_all_attacks": len(shared_all),
    "fgms_shared_by_all_attacks": fmt(shared_all),
    "n_attack_specific_fgms_total": sum(len(s) for s in unique_any.values()),

    "mean_text_similarity_to_unconstrained": alert_out[
        "constrained_vs_unconstrained_text_similarity"
    ].mean(),

    "human_fgm_overlap_rate": alert_out.loc[
        alert_out["has_human_reference"],
        "constrained_human_fgm_overlap"
    ].mean(),

    "mean_shared_fgms_with_human": alert_out.loc[
        alert_out["has_human_reference"],
        "n_constrained_human_shared_fgms"
    ].mean(),

    "judge_fgm_overlap_rate": alert_out.loc[
        alert_out["has_llm_judge_reference"],
        "constrained_judge_fgm_overlap"
    ].mean(),

    "mean_shared_fgms_with_judge": alert_out.loc[
        alert_out["has_llm_judge_reference"],
        "n_constrained_judge_shared_fgms"
    ].mean(),
}

summary_out = pd.DataFrame([summary])
summary_out.to_csv(OUT_SUMMARY, index=False)


# ================= PRINT OUTPUT =================

print("\nDONE")
print(f"Alert-level evaluation: {OUT_ALERT_LEVEL}")
print(f"Attack-level evaluation: {OUT_ATTACK_LEVEL}")
print(f"Invalid FGMs: {OUT_INVALID_FGMS}")
print(f"FGM frequency: {OUT_FGM_FREQ}")
print(f"Attack specificity: {OUT_ATTACK_SPECIFICITY}")
print(f"Summary: {OUT_SUMMARY}")

print("\nSummary:")
print(summary_out.T)

print("\nTop FGMs:")
print(freq_out.head(15))

print("\nAttack-specific FGM analysis:")
print(attack_specificity_out)
import re
from pathlib import Path
import pandas as pd

# ================= CONFIG =================

HUMAN_CSV = Path("validation/normalized_all_validators.csv")
LLM_CSV = Path("llm_judge_unconstrained/llm_judge_long_format.csv")

OUT_DIR = Path("validation_comparison")
OUT_DIR.mkdir(exist_ok=True)

OUT_COMPARISON = OUT_DIR / "human_vs_llm_comparison.csv"
OUT_ATTACK_SUMMARY = OUT_DIR / "agreement_by_attack_class.csv"
OUT_SUMMARY = OUT_DIR / "comparison_summary.csv"
OUT_TIME = OUT_DIR / "time_comparison.csv"


# ================= HELPERS =================

def clean_id(x):
    if pd.isna(x):
        return ""

    x = str(x).strip().upper()

    if x in ["", "NAN", "NONE", "NULL"]:
        return ""

    return x


def canonical_attack(x):
    if pd.isna(x):
        return ""

    s = str(x).lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    if "bidding" in s and "service" in s:
        return "bidding_down_service_reject"
    if "location" in s:
        return "location_tracking"
    if "rrc" in s and "replay" in s:
        return "rrc_replay"
    if "energy" in s:
        return "energy_depletion"
    if "mobile network" in s or "mnmap" in s:
        return "mobile_network_mapping"
    if "paging" in s:
        return "paging_hijacking"

    return s


def extract_first_number(x):
    if pd.isna(x):
        return ""

    match = re.search(r"\d+", str(x))
    return str(int(match.group(0))) if match else ""


def extract_final_number(x):
    if pd.isna(x):
        return ""

    match = re.search(r"(\d+)$", str(x))
    return str(int(match.group(1))) if match else ""


def normalize_fgm_cell(x):
    x = clean_id(x)

    if not x:
        return tuple()

    if x in ["NO_MATCH", "UNKNOWN", "N/A", "NA"]:
        return (x,)

    parts = x.replace(",", ";").replace("|", ";").split(";")
    parts = [clean_id(p) for p in parts if clean_id(p)]

    return tuple(sorted(set(parts)))


def exact_set_match(a, b):
    return normalize_fgm_cell(a) == normalize_fgm_cell(b)


def overlap_match(a, b):
    a_set = set(normalize_fgm_cell(a))
    b_set = set(normalize_fgm_cell(b))

    if not a_set and not b_set:
        return True

    if not a_set or not b_set:
        return False

    return len(a_set & b_set) > 0


def no_match_rate(series):
    return series.isin(["NO_MATCH", "UNKNOWN", "N/A", "NA", ""]).mean()


def ensure_columns(df, source, required):
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


# ================= LOAD =================

human = pd.read_csv(HUMAN_CSV)
llm = pd.read_csv(LLM_CSV)

ensure_columns(
    human,
    "human CSV",
    [
        "attack_key",
        "alert_id",
        "attack_label",
        "validator",
        "fight_technique_id",
        "mitigation_index",
        "mitigation_text",
        "mapped_fgm_ids",
        "time_taken_minutes",
    ],
)

ensure_columns(
    llm,
    "LLM CSV",
    [
        "sample_id",
        "attack_label",
        "fight_technique_id",
        "mitigation_index",
        "mitigation_text",
        "mapped_fgm_ids",
        "time_taken_minutes",
    ],
)


# ================= NORMALIZE KEYS =================

human["attack_key"] = human["attack_key"].apply(canonical_attack)
human["alert_id"] = human["alert_id"].apply(extract_first_number)

llm["attack_key"] = llm["sample_id"].apply(canonical_attack)
llm["alert_id"] = llm["sample_id"].apply(extract_final_number)

for df in [human, llm]:
    df["attack_key"] = df["attack_key"].astype(str)
    df["alert_id"] = df["alert_id"].astype(str)

    df["mitigation_index"] = (
        pd.to_numeric(df["mitigation_index"], errors="coerce")
        .fillna(1)
        .astype(int)
    )

    df["fight_technique_id"] = df["fight_technique_id"].apply(clean_id)
    df["mapped_fgm_ids"] = df["mapped_fgm_ids"].apply(
        lambda x: ";".join(normalize_fgm_cell(x))
    )


# ================= COMPARE =================

human_ref = human.rename(
    columns={
        "attack_label": "human_attack_label",
        "validator": "human_validator",
        "fight_technique_id": "human_fgt",
        "mitigation_text": "human_mitigation_text",
        "mapped_fgm_ids": "human_fgm",
        "time_taken_minutes": "human_time_taken_minutes",
    }
)

llm_ref = llm.rename(
    columns={
        "attack_label": "llm_attack_label",
        "fight_technique_id": "llm_fgt",
        "mitigation_text": "llm_mitigation_text",
        "mapped_fgm_ids": "llm_fgm",
        "time_taken_minutes": "llm_time_taken_minutes",
    }
)

comparison = human_ref.merge(
    llm_ref,
    on=["attack_key", "alert_id", "mitigation_index"],
    how="inner",
)

comparison["fgt_match"] = comparison["human_fgt"] == comparison["llm_fgt"]

comparison["fgm_exact_match"] = comparison.apply(
    lambda r: exact_set_match(r["human_fgm"], r["llm_fgm"]),
    axis=1,
)

comparison["fgm_overlap_match"] = comparison.apply(
    lambda r: overlap_match(r["human_fgm"], r["llm_fgm"]),
    axis=1,
)

comparison.to_csv(OUT_COMPARISON, index=False)


# ================= ATTACK-LEVEL SUMMARY =================

attack_summary = (
    comparison.groupby("attack_key")
    .agg(
        n_comparisons=("attack_key", "size"),
        fgt_agreement=("fgt_match", "mean"),
        fgm_exact_agreement=("fgm_exact_match", "mean"),
        fgm_overlap_agreement=("fgm_overlap_match", "mean"),
    )
    .reset_index()
)

attack_summary.to_csv(OUT_ATTACK_SUMMARY, index=False)


# ================= OVERALL SUMMARY =================

human_pairs = human[["attack_key", "alert_id"]].drop_duplicates()
llm_pairs = llm[["attack_key", "alert_id"]].drop_duplicates()

shared_pairs = human_pairs.merge(
    llm_pairs,
    on=["attack_key", "alert_id"],
    how="inner",
)

summary = {
    "n_human_rows": len(human),
    "n_llm_rows": len(llm),
    "n_human_attack_alert_pairs": len(human_pairs),
    "n_llm_attack_alert_pairs": len(llm_pairs),
    "n_shared_attack_alert_pairs": len(shared_pairs),

    "n_comparisons": len(comparison),
    "fgt_agreement": comparison["fgt_match"].mean() if len(comparison) else None,
    "fgm_exact_agreement": comparison["fgm_exact_match"].mean() if len(comparison) else None,
    "fgm_overlap_agreement": comparison["fgm_overlap_match"].mean() if len(comparison) else None,

    "human_no_match_rate": no_match_rate(human["mapped_fgm_ids"]),
    "llm_no_match_rate": no_match_rate(llm["mapped_fgm_ids"]),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUT_SUMMARY, index=False)


# ================= TIME COMPARISON =================

human_time = (
    human.groupby("validator")["time_taken_minutes"]
    .max()
    .reset_index()
    .rename(columns={"time_taken_minutes": "total_time_minutes"})
)

llm_time = pd.DataFrame([
    {
        "validator": "LLM_JUDGE",
        "total_time_minutes": llm["time_taken_minutes"].max(),
    }
])

time_comparison = pd.concat([human_time, llm_time], ignore_index=True)
time_comparison.to_csv(OUT_TIME, index=False)


# ================= PRINT =================

print("\nDONE")
print(f"Comparison: {OUT_COMPARISON}")
print(f"Attack summary: {OUT_ATTACK_SUMMARY}")
print(f"Overall summary: {OUT_SUMMARY}")
print(f"Time comparison: {OUT_TIME}")

print("\nOverall summary:")
print(summary_df.T)

print("\nAgreement by attack class:")
print(attack_summary)
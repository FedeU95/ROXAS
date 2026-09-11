from pathlib import Path
import re
import json
import pandas as pd

INPUT_FOLDER = Path("validation/outputs")
OUTPUT_CSV = "normalized_all_validators.csv"
OUTPUT_XLSX = "normalized_all_validators.xlsx"
SUMMARY_JSON = "normalization_summary.json"


def clean_col(name):
    return re.sub(r"\s+", " ", str(name).replace("\n", " ")).strip().lower()


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


def read_file(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def find_col(df, keywords, exclude=None):
    exclude = exclude or []
    for c in df.columns:
        cc = clean_col(c)
        if any(e in cc for e in exclude):
            continue
        if all(k in cc for k in keywords):
            return c
    return None


def find_validator_col(df):
    for c in df.columns:
        if clean_col(c) in ["validator", "participant", "human validator"]:
            return c
    return None


def find_alert_col(df):
    for c in df.columns:
        cc = clean_col(c)
        if "alert" in cc and "id" in cc:
            return c
    return None


def extract_alert_id(value):
    if pd.isna(value):
        return ""
    match = re.search(r"\d+", str(value))
    return str(int(match.group(0))) if match else ""


def parse_minutes(value):
    if pd.isna(value):
        return None

    text = str(value).lower()
    total = 0

    h = re.search(r"(\d+)\s*h(?:ours?)?", text)
    m = re.search(r"(\d+)\s*m(?:in(?:ute)?s?)?", text)

    if h:
        total += int(h.group(1)) * 60
    if m:
        total += int(m.group(1))

    if total > 0:
        return total

    nums = re.findall(r"\d+", text)
    return int(nums[0]) if nums else None


def extract_fgms(value):
    if pd.isna(value):
        return ""

    fgms = re.findall(r"FGM\d+", str(value).upper())
    return ";".join(sorted(set(fgms)))


def split_numbered_items(value):
    if pd.isna(value):
        return []

    text = str(value).strip()
    matches = list(re.finditer(r"(?m)^\s*\d+[.)]\s*", text))

    if not matches:
        return [text] if text else []

    items = []

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item = text[start:end].strip()
        if item:
            items.append(item)

    return items


def is_packed_cell(value):
    if pd.isna(value):
        return False

    text = str(value)
    return (
        bool(re.search(r"(?m)^\s*1[.)]\s*", text))
        and bool(re.search(r"(?m)^\s*2[.)]\s*", text))
    )


def normalize_file(path):
    df = read_file(path)
    df.columns = [clean_col(c) for c in df.columns]

    validator_col = find_validator_col(df)
    alert_col = find_alert_col(df)
    attack_col = find_col(df, ["attack"], exclude=["mitigation"])
    mitigation_col = find_col(df, ["mitigation"], exclude=["id", "technique"])
    fgm_col = find_col(df, ["mitigation", "id"])
    fgt_col = find_col(df, ["technique", "id"])
    time_col = find_col(df, ["time"])

    if mitigation_col is None:
        raise ValueError(f"Could not find mitigation column in {path.name}")

    if alert_col is None:
        raise ValueError(f"Could not find alert ID column in {path.name}")

    if validator_col:
        df[validator_col] = df[validator_col].ffill()

    rows = []

    for _, row in df.iterrows():
        mitigation_value = row.get(mitigation_col)

        if pd.isna(mitigation_value):
            continue

        validator = row.get(validator_col) if validator_col else path.stem
        alert_id = extract_alert_id(row.get(alert_col))
        attack = row.get(attack_col) if attack_col else None
        attack_key = canonical_attack(attack)
        fgt = row.get(fgt_col) if fgt_col else None
        time_taken = parse_minutes(row.get(time_col)) if time_col else None
        fgm_value = row.get(fgm_col) if fgm_col else None

        if is_packed_cell(mitigation_value):
            mitigations = split_numbered_items(mitigation_value)
            fgms = split_numbered_items(fgm_value)

            max_len = max(len(mitigations), len(fgms))

            for i in range(max_len):
                rows.append({
                    "attack_key": attack_key,
                    "alert_id": alert_id,
                    "mitigation_index": i + 1,
                    "attack_label": attack,
                    "validator": validator,
                    "fight_technique_id": fgt,
                    "mitigation_text": mitigations[i] if i < len(mitigations) else "",
                    "mapped_fgm_ids": extract_fgms(fgms[i] if i < len(fgms) else ""),
                    "time_taken_minutes": time_taken,
                    "source_file": path.name,
                    "source_format": "packed",
                })

        else:
            rows.append({
                "attack_key": attack_key,
                "alert_id": alert_id,
                "mitigation_index": None,
                "attack_label": attack,
                "validator": validator,
                "fight_technique_id": fgt,
                "mitigation_text": str(mitigation_value).strip(),
                "mapped_fgm_ids": extract_fgms(fgm_value),
                "time_taken_minutes": time_taken,
                "source_file": path.name,
                "source_format": "rowwise",
            })

    return rows


def main():
    files = (
        list(INPUT_FOLDER.glob("*.csv"))
        + list(INPUT_FOLDER.glob("*.xlsx"))
        + list(INPUT_FOLDER.glob("*.xls"))
    )

    if not files:
        raise FileNotFoundError(f"No files found in {INPUT_FOLDER}")

    all_rows = []
    summary = {}

    for path in sorted(files):
        try:
            rows = normalize_file(path)
            all_rows.extend(rows)

            summary[path.name] = {
                "status": "success",
                "rows_generated": len(rows),
            }

            print(f"OK: {path.name} -> {len(rows)} rows")

        except Exception as e:
            summary[path.name] = {
                "status": "failed",
                "error": str(e),
            }

            print(f"FAILED: {path.name} -> {e}")

    out = pd.DataFrame(all_rows)

    if out.empty:
        raise RuntimeError("No rows parsed")

    # Re-index mitigations per validator AND per attack+alert.
    out["mitigation_index"] = (
        out.groupby(["attack_key", "alert_id", "validator"])
        .cumcount()
        .add(1)
    )

    # Sum time once per validator/source file.
    time_per_file = (
        out[["validator", "source_file", "time_taken_minutes"]]
        .dropna()
        .drop_duplicates()
    )

    validator_total_time = (
        time_per_file.groupby("validator")["time_taken_minutes"]
        .sum()
        .to_dict()
    )

    out["time_taken_minutes"] = out["validator"].map(validator_total_time)

    # Organize by attack and alert, not validator.
    out = out[
        [
            "attack_key",
            "alert_id",
            "mitigation_index",
            "attack_label",
            "validator",
            "fight_technique_id",
            "mitigation_text",
            "mapped_fgm_ids",
            "time_taken_minutes",
            "source_file",
            "source_format",
        ]
    ]

    out = out.sort_values(
        by=["attack_key", "alert_id", "mitigation_index", "validator"],
        key=lambda col: pd.to_numeric(col, errors="ignore")
    )

    out.to_csv(OUTPUT_CSV, index=False)
    out.to_excel(OUTPUT_XLSX, index=False)

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Excel: {OUTPUT_XLSX}")
    print(f"Summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
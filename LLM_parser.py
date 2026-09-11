import re
import csv
from pathlib import Path

RAW_DIR = Path("llm_mitigation_exp_constrained")
OUT_CSV = Path("llm_mitigation_exp_constrained/llm_constrained_long_format.csv")


def canonical_attack(x):
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
    if "mobile network" in s or "mnmap" in s:
        return "mobile_network_mapping"
    if "paging" in s:
        return "paging_hijacking"

    return s


def infer_attack_label(sample_id):
    s = sample_id.lower()

    if s.startswith("bidding_down_with_servicereject"):
        return "Bidding Down with Service Reject"
    if s.startswith("location_tracking"):
        return "Location Tracking"
    if s.startswith("rrc_replay_attack"):
        return "RRC Replay"
    if s.startswith("energy_depletion_attack"):
        return "Energy Depletion"
    if s.startswith("mobile_network_mapping"):
        return "Mobile Network Mapping"
    if s.startswith("paging_channel_hijacking"):
        return "Paging Capability Hijacking"

    return ""


def extract_alert_id(sample_id):
    m = re.search(r"(\d+)$", sample_id)
    return str(int(m.group(1))) if m else ""


def extract_fgm_ids(text):
    return sorted(set(re.findall(r"FGM\d+(?:\.\d+)?", text.upper())))


def parse_response_file(path):
    text = path.read_text(encoding="utf-8", errors="replace")

    sample_id = path.name.replace("_response.txt", "")
    attack_label = infer_attack_label(sample_id)
    attack_key = canonical_attack(sample_id)
    alert_id = extract_alert_id(sample_id)

    rows = []
    mitigation_index = 0

    for line in text.splitlines():
        raw = line.strip()

        if not raw:
            continue

        cleaned = re.sub(r"^\s*[-*•]\s*", "", raw)
        cleaned = re.sub(r"^\s*\d+[.)]\s*", "", cleaned).strip()

        fgms = extract_fgm_ids(cleaned)

        if not fgms:
            continue

        mitigation_index += 1

        rows.append({
            "validator": "LLM_CONSTRAINED_open-mistral-7b",
            "sample_id": sample_id,
            "attack_key": attack_key,
            "alert_id": alert_id,
            "attack_label": attack_label,
            "fight_technique_id": "",
            "mitigation_index": mitigation_index,
            "mitigation_text": cleaned,
            "mapped_fgm_ids": ";".join(fgms),
            "time_taken_minutes": "",
            "source": "llm_constrained",
            "source_file": path.name,
        })

    return rows


def main():
    files = sorted(RAW_DIR.glob("*_response.txt"))

    if not files:
        raise FileNotFoundError(f"No *_response.txt files found in {RAW_DIR}")

    all_rows = []

    for path in files:
        rows = parse_response_file(path)
        all_rows.extend(rows)
        print(f"{path.name}: parsed {len(rows)} mitigations")

    fieldnames = [
        "validator",
        "sample_id",
        "attack_key",
        "alert_id",
        "attack_label",
        "fight_technique_id",
        "mitigation_index",
        "mitigation_text",
        "mapped_fgm_ids",
        "time_taken_minutes",
        "source",
        "source_file",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Wrote {len(all_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
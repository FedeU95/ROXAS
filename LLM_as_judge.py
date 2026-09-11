import os
import time
import json
import random
import csv
from typing import List, Dict, Any

# ================= CONFIG =================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not set")

from mistralai import Mistral

client = Mistral(api_key=MISTRAL_API_KEY)

MISTRAL_MODEL = "open-mistral-7b"

TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1200
MIN_MAX_TOKENS = 128
SLEEP_BETWEEN_CALLS = 2

PROMPT_DIR = "prompts_new"
UNCONSTRAINED_DIR = "llm_mitigation_exp"
CONTEXT_FILE = "context_mit.txt"

FIGHT_DB_DIR = "fight_db"
FIGHT_TECHNIQUES_FILE = os.path.join(FIGHT_DB_DIR, "fight_techniques.json")
FIGHT_MITIGATIONS_FILE = os.path.join(FIGHT_DB_DIR, "fight_mitigations.json")

JUDGE_OUT = "llm_judge_unconstrained"
RAW_DIR = os.path.join(JUDGE_OUT, "raw")
PARSED_DIR = os.path.join(JUDGE_OUT, "parsed")
GROUPED_DIR = os.path.join(JUDGE_OUT, "grouped")

TIMING_FILE = os.path.join(JUDGE_OUT, "timing_summary.json")
LONG_CSV = os.path.join(JUDGE_OUT, "llm_judge_long_format.csv")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PARSED_DIR, exist_ok=True)
os.makedirs(GROUPED_DIR, exist_ok=True)


# ================= LOAD CONTEXT =================

with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
    initial_context = f.read().strip()


# ================= LOAD FiGHT DB =================

def safe_get(d: dict, *keys, default=""):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return default


def load_fight_db():
    with open(FIGHT_TECHNIQUES_FILE, "r", encoding="utf-8") as f:
        techniques = json.load(f)

    with open(FIGHT_MITIGATIONS_FILE, "r", encoding="utf-8") as f:
        mitigations = json.load(f)

    return techniques, mitigations


def build_fight_context_text(techniques, mitigations) -> str:
    lines = []

    lines.append("MITRE FiGHT reference database")
    lines.append("")
    lines.append("Techniques:")
    for t in techniques:
        tech_id = safe_get(t, "technique_id", "id", default="UNKNOWN")
        tech_name = safe_get(t, "name", "title", default="UNKNOWN")
        lines.append(f"- {tech_id}: {tech_name}")

    lines.append("")
    lines.append("Mitigations:")
    for m in mitigations:
        mit_id = safe_get(m, "mitigation_id", "id", default="UNKNOWN")
        mit_name = safe_get(m, "name", "title", default="UNKNOWN")
        lines.append(f"- {mit_id}: {mit_name}")

    return "\n".join(lines)


fight_techniques, fight_mitigations = load_fight_db()
fight_context = build_fight_context_text(fight_techniques, fight_mitigations)


# ================= ATTACK LABEL FROM FILENAME =================

def infer_attack_label_from_filename(fname: str) -> str:
    name = fname.lower()

    if name.startswith("energy_depletion_attack_"):
        return "Energy Depletion"
    elif name.startswith("mobile_network_mapping_(mnmap)_"):
        return "Mobile Network Mapping"
    elif name.startswith("paging_channel_hijacking_attack_"):
        return "Paging Capability Hijacking"
    elif name.startswith("rrc_replay_attack_"):
        return "RRC Replay"
    elif name.startswith("bidding_down_with_servicereject_"):
        return "Bidding Down with Service Reject"
    elif name.startswith("location_tracking_"):
        return "Location Tracking"
    else:
        raise ValueError(f"Unknown attack label in filename: {fname}")


# ================= JUDGE PROMPT =================

def build_judge_messages(
    attack_features: str,
    llm_output: str,
    attack_label: str,
    sample_id: str,
    fight_context: str
):
    system_msg = (
        "You are an expert evaluator for 5G network security mitigation actions.\n"
        "Use the provided MITRE FiGHT reference database as the source of truth.\n"
        "Your job is to perform the same mapping task as a human validator.\n"
        "Do not invent technique IDs or mitigation IDs.\n"
        "Return JSON only."
    )

    user_msg = (
        f"{fight_context}\n\n"
        f"Sample ID: {sample_id}\n\n"
        f"Context:\n{initial_context}\n\n"
        f"Ground-truth attack label:\n{attack_label}\n\n"
        f"Most anomalous feature-value pairs:\n{attack_features}\n\n"
        f"Mitigation output to evaluate:\n{llm_output}\n\n"
        "Task:\n"
        "1) Map the ground-truth attack label to the single most relevant MITRE FiGHT technique ID.\n"
        "2) Extract the mitigation actions as an ordered list.\n"
        "3) Map each action to the closest MITRE FiGHT mitigation ID.\n\n"
        "Return JSON with exactly this structure:\n"
        "{\n"
        '  "sample_id": "<sample id>",\n'
        '  "attack_label": "<ground truth label>",\n'
        '  "fight_technique_id": "<FGT id or UNKNOWN>",\n'
        '  "fight_technique_name": "<technique name or UNKNOWN>",\n'
        '  "actions": [\n'
        "    {\n"
        '      "index": 1,\n'
        '      "action_text": "<action text>",\n'
        '      "fgm_id": "<FGM id or NO_MATCH or UNKNOWN>",\n'
        '      "fgm_name": "<mitigation name or NO_MATCH or UNKNOWN>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Preserve the original mitigation order.\n"
        "- Include one object per mitigation action.\n"
        "- Do not merge actions.\n"
        "- Do not omit actions.\n"
        "- Use only IDs present in the FiGHT reference above.\n"
        "- If a generic action such as IDS/IPS deployment does not clearly match a FiGHT mitigation, use NO_MATCH.\n"
        "- If the correct ID cannot be determined from the provided FiGHT reference, use UNKNOWN.\n"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


# ================= API CALL =================

def mistral_chat_complete(
    messages: List[Dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = TEMPERATURE,
    max_retries: int = 6
) -> str:
    max_tokens = max(MIN_MAX_TOKENS, max_tokens)

    for attempt in range(max_retries):
        try:
            resp = client.chat.complete(
                model=MISTRAL_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

            if not resp or not resp.choices:
                raise RuntimeError("Empty response")

            text = resp.choices[0].message.content or ""
            return text.strip()

        except Exception as e:
            emsg = str(e).lower()

            retryable = any(x in emsg for x in [
                "429",
                "rate limit",
                "timeout",
                "timed out",
                "503",
                "service unavailable",
                "bad gateway",
                "502"
            ])

            if attempt < max_retries - 1 and retryable:
                backoff = (2 ** attempt) + random.uniform(0, 0.5)
                print(f"[!] Retryable error on attempt {attempt + 1}/{max_retries}: {e}")
                print(f"    Sleeping {backoff:.2f}s before retry...")
                time.sleep(backoff)
                continue

            raise


# ================= JSON PARSER =================

def extract_json(text: str) -> Any:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            return json.loads(candidate)

    raise ValueError("No valid JSON found in response")


# ================= POST-PROCESSING =================

def normalize_actions(parsed: dict) -> dict:
    actions = parsed.get("actions", [])

    if not isinstance(actions, list):
        actions = []

    normalized = []

    for item in actions:
        if not isinstance(item, dict):
            continue

        idx = item.get("index")

        try:
            idx = int(idx)
        except Exception:
            idx = len(normalized) + 1

        normalized.append({
            "index": idx,
            "action_text": str(item.get("action_text", "")).strip(),
            "fgm_id": str(item.get("fgm_id", "UNKNOWN")).strip(),
            "fgm_name": str(item.get("fgm_name", "UNKNOWN")).strip(),
        })

    normalized.sort(key=lambda x: x["index"])

    parsed["actions"] = normalized
    return parsed


def group_actions_by_fgm(parsed: dict) -> dict:
    grouped = {}
    ordered_actions = []

    for item in parsed.get("actions", []):
        idx = item.get("index")
        txt = item.get("action_text", "")
        fgm_id = item.get("fgm_id", "UNKNOWN")
        fgm_name = item.get("fgm_name", "UNKNOWN")

        ordered_actions.append(txt)

        key = (fgm_id, fgm_name)
        grouped.setdefault(key, []).append(idx)

    mapped_mitigations = []

    for (fgm_id, fgm_name), indices in grouped.items():
        mapped_mitigations.append({
            "fgm_id": fgm_id,
            "fgm_name": fgm_name,
            "action_indices": sorted(indices)
        })

    mapped_mitigations.sort(
        key=lambda x: min(x.get("action_indices", [9999]))
    )

    return {
        "sample_id": parsed.get("sample_id"),
        "attack_label": parsed.get("attack_label"),
        "fight_technique_id": parsed.get("fight_technique_id"),
        "fight_technique_name": parsed.get("fight_technique_name"),
        "actions": ordered_actions,
        "mapped_mitigations": mapped_mitigations
    }


def parsed_to_long_rows(parsed: dict, elapsed_minutes: float) -> List[dict]:
    rows = []

    for action in parsed.get("actions", []):
        rows.append({
            "validator": f"LLM_JUDGE_{MISTRAL_MODEL}",
            "sample_id": parsed.get("sample_id"),
            "attack_label": parsed.get("attack_label"),
            "fight_technique_id": parsed.get("fight_technique_id"),
            "mitigation_index": action.get("index"),
            "mitigation_text": action.get("action_text"),
            "mapped_fgm_ids": action.get("fgm_id"),
            "mapped_fgm_names": action.get("fgm_name"),
            "time_taken_minutes": elapsed_minutes,
            "source": "llm_judge",
        })

    return rows


# ================= RUNNER =================

def main():
    run_start_time = time.time()

    timing_summary = []
    long_rows = []

    for fname in sorted(os.listdir(PROMPT_DIR)):
        if not fname.endswith(".txt"):
            continue

        sample_start_time = time.time()

        prompt_path = os.path.join(PROMPT_DIR, fname)
        response_path = os.path.join(
            UNCONSTRAINED_DIR,
            fname.replace(".txt", "_response.txt")
        )

        if not os.path.exists(response_path):
            print(f"[!] Missing LLM output for {fname}")
            timing_summary.append({
                "sample_id": fname.replace(".txt", ""),
                "attack_label": None,
                "elapsed_seconds": 0,
                "elapsed_minutes": 0,
                "status": "missing_llm_output",
            })
            continue

        try:
            attack_label = infer_attack_label_from_filename(fname)
        except ValueError as e:
            print(f"[!] {e}")
            timing_summary.append({
                "sample_id": fname.replace(".txt", ""),
                "attack_label": None,
                "elapsed_seconds": 0,
                "elapsed_minutes": 0,
                "status": "unknown_attack_label",
                "error": str(e),
            })
            continue

        sample_id = fname.replace(".txt", "")

        raw_out = os.path.join(RAW_DIR, sample_id + "_judge_raw.txt")
        parsed_out = os.path.join(PARSED_DIR, sample_id + "_judge.json")
        grouped_out = os.path.join(GROUPED_DIR, sample_id + "_judge_grouped.json")

        with open(prompt_path, "r", encoding="utf-8") as f:
            attack_features = f.read().strip()

        with open(response_path, "r", encoding="utf-8") as f:
            llm_output = f.read().strip()

        messages = build_judge_messages(
            attack_features=attack_features,
            llm_output=llm_output,
            attack_label=attack_label,
            sample_id=sample_id,
            fight_context=fight_context
        )

        try:
            print(f"\nJudging {fname} | {attack_label}")

            judge_text = mistral_chat_complete(
                messages=messages,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=TEMPERATURE
            )

            with open(raw_out, "w", encoding="utf-8") as f:
                f.write(judge_text)

            parsed = extract_json(judge_text)
            parsed = normalize_actions(parsed)

            elapsed_seconds = time.time() - sample_start_time
            elapsed_minutes = elapsed_seconds / 60

            with open(parsed_out, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

            grouped = group_actions_by_fgm(parsed)

            with open(grouped_out, "w", encoding="utf-8") as f:
                json.dump(grouped, f, indent=2, ensure_ascii=False)

            long_rows.extend(parsed_to_long_rows(parsed, elapsed_minutes))

            timing_summary.append({
                "sample_id": sample_id,
                "attack_label": attack_label,
                "elapsed_seconds": elapsed_seconds,
                "elapsed_minutes": elapsed_minutes,
                "status": "success",
            })

            print(f"Saved parsed judge output -> {parsed_out}")
            print(f"Saved grouped judge output -> {grouped_out}")
            print(f"Elapsed: {elapsed_seconds:.2f}s")

            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            elapsed_seconds = time.time() - sample_start_time
            elapsed_minutes = elapsed_seconds / 60

            timing_summary.append({
                "sample_id": sample_id,
                "attack_label": attack_label,
                "elapsed_seconds": elapsed_seconds,
                "elapsed_minutes": elapsed_minutes,
                "status": "failed",
                "error": str(e),
            })

            print(f"[!] Failed on {fname}: {e}")
            print(f"    Raw output path: {raw_out}")
            continue

    total_elapsed_seconds = time.time() - run_start_time
    total_elapsed_minutes = total_elapsed_seconds / 60

    with open(TIMING_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "model": MISTRAL_MODEL,
            "total_elapsed_seconds": total_elapsed_seconds,
            "total_elapsed_minutes": total_elapsed_minutes,
            "samples": timing_summary,
        }, f, indent=2, ensure_ascii=False)

    if long_rows:
        with open(LONG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
            writer.writeheader()
            writer.writerows(long_rows)

    print("\nDONE")
    print(f"Timing summary -> {TIMING_FILE}")
    print(f"Long-format CSV -> {LONG_CSV}")


if __name__ == "__main__":
    main()
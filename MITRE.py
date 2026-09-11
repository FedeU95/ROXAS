import os
import json
import requests
import yaml

OUT_DIR = "fight_db"
os.makedirs(OUT_DIR, exist_ok=True)

# Official public FiGHT repo / raw YAML
RAW_YAML_URL = "https://raw.githubusercontent.com/mitre/FiGHT/main/fight.yaml"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ROXAS-FIGHT-Downloader/1.0)"
}


def download_fight_yaml() -> dict:
    r = requests.get(RAW_YAML_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = yaml.safe_load(r.text)
    if not isinstance(data, dict):
        raise ValueError("Unexpected YAML top-level structure")
    return data


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_yaml_text(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def extract_objects(root):
    """
    Tries to find technique-like and mitigation-like objects recursively.
    This is schema-tolerant because the exact FiGHT YAML structure may evolve.
    """
    techniques = []
    mitigations = []

    def walk(node):
        if isinstance(node, dict):
            # Technique-like object
            tech_id = None
            mit_id = None

            for k, v in node.items():
                if isinstance(v, str):
                    if v.startswith("FGT"):
                        tech_id = v
                    elif v.startswith("FGM"):
                        mit_id = v

            if tech_id:
                techniques.append(node)
            if mit_id:
                mitigations.append(node)

            for v in node.values():
                walk(v)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(root)

    # Deduplicate by first FGT/FGM string found
    def dedupe(objs, prefix):
        seen = set()
        out = []
        for obj in objs:
            obj_id = None
            for v in obj.values():
                if isinstance(v, str) and v.startswith(prefix):
                    obj_id = v
                    break
            if obj_id and obj_id not in seen:
                seen.add(obj_id)
                out.append(obj)
        return out

    return dedupe(techniques, "FGT"), dedupe(mitigations, "FGM")


def normalize_techniques(raw_techniques):
    out = []
    for obj in raw_techniques:
        tech_id = next((v for v in obj.values() if isinstance(v, str) and v.startswith("FGT")), None)
        name = obj.get("name") or obj.get("title") or obj.get("label") or ""
        summary = (
            obj.get("summary")
            or obj.get("description")
            or obj.get("desc")
            or ""
        )
        out.append({
            "technique_id": tech_id,
            "name": name,
            "summary": summary,
            "raw": obj
        })
    return sorted(out, key=lambda x: x["technique_id"] or "")


def normalize_mitigations(raw_mitigations):
    out = []
    for obj in raw_mitigations:
        mit_id = next((v for v in obj.values() if isinstance(v, str) and v.startswith("FGM")), None)
        name = obj.get("name") or obj.get("title") or obj.get("label") or ""
        summary = (
            obj.get("summary")
            or obj.get("description")
            or obj.get("desc")
            or ""
        )
        out.append({
            "mitigation_id": mit_id,
            "name": name,
            "summary": summary,
            "raw": obj
        })
    return sorted(out, key=lambda x: x["mitigation_id"] or "")


def main():
    print("Downloading FiGHT YAML...")
    r = requests.get(RAW_YAML_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    raw_text = r.text
    save_yaml_text(raw_text, os.path.join(OUT_DIR, "fight.yaml"))

    data = yaml.safe_load(raw_text)
    print("YAML loaded.")

    # Save full parsed YAML
    save_json(data, os.path.join(OUT_DIR, "fight_full.json"))

    raw_techniques, raw_mitigations = extract_objects(data)
    techniques = normalize_techniques(raw_techniques)
    mitigations = normalize_mitigations(raw_mitigations)

    save_json(techniques, os.path.join(OUT_DIR, "fight_techniques.json"))
    save_json(mitigations, os.path.join(OUT_DIR, "fight_mitigations.json"))

    print(f"Saved YAML to {os.path.join(OUT_DIR, 'fight.yaml')}")
    print(f"Saved full JSON to {os.path.join(OUT_DIR, 'fight_full.json')}")
    print(f"Saved techniques to {os.path.join(OUT_DIR, 'fight_techniques.json')} ({len(techniques)})")
    print(f"Saved mitigations to {os.path.join(OUT_DIR, 'fight_mitigations.json')} ({len(mitigations)})")


if __name__ == "__main__":
    main()
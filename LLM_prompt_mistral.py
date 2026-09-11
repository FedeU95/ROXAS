import os
import time
import random
from time import perf_counter
from typing import List, Dict
import pandas as pd

# ============ CONFIG (Mistral API) ============
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not set in environment")

MISTRAL_MODEL = "open-mistral-7b"

SLEEP_BETWEEN_CALLS = 1
USE_EXAMPLES = False
PROMPT_DIR = "prompts_new"
BASE_OUT = "llm_mitigation_exp"
RESPONSE_DIR = os.path.join(BASE_OUT)
os.makedirs(RESPONSE_DIR, exist_ok=True)

DEFAULT_MAX_TOKENS = 1024
MIN_MAX_TOKENS = 64
TEMPERATURE = 0.0

# ============ Load context ============
context_file = "context_mit.txt"
with open(context_file, "r") as f:
    initial_context = f.read().strip()

# ============ Mistral client ============
from mistralai import Mistral

client = Mistral(api_key=MISTRAL_API_KEY)

# ============ Messaging ============
def build_messages(attack_features: str):
    msgs = [
        {
            "role": "system",
            "content": "You are a cybersecurity analyst specializing in wireless network attacks."
        },
        {
            "role": "user",
            "content": (
                f"{initial_context}\n\n"
                f"Most anomalous feature–value pairs:\n{attack_features}\n\n"
                "Instructions (must follow exactly):\n"
                "- Output exactly TWO sections only, in plain text:\n"
                "  1) Cause\n"
                "  2) Mitigations\n"
                "- Do NOT use markdown, headings, numbering, or sub-bullets.\n"
                "- Cause: exactly 1 sentence, total ≤ 70 words. "
                "Include assumptions inline only if necessary.\n"
                "- Mitigations: up to 10 Bullets, each one line, ≤ 12 words each.\n"
                "- Each mitigation bullet may include timer OR counter OR KPI OR rule.\n"
                "- Restrict content to 5G NAS/RRC and core/RAN configuration.\n"
            )
        }
    ]
    return msgs

# ============ API call with retry/backoff ============
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
            )

            if not resp or not resp.choices:
                raise RuntimeError("Empty response from Mistral API")

            text = resp.choices[0].message.content or ""
            return text.strip()

        except Exception as e:
            emsg = str(e).lower()

            retryable = any(x in emsg for x in [
                "rate limit", "429", "timeout", "timed out",
                "service unavailable", "503", "bad gateway", "502"
            ])

            if attempt < max_retries - 1 and retryable:
                backoff = (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(backoff)
                continue

            raise

# ============ Runner ============
runtime_rows = []

for fname in sorted(os.listdir(PROMPT_DIR)):
    if not fname.endswith(".txt"):
        continue

    prompt_path = os.path.join(PROMPT_DIR, fname)
    response_path = os.path.join(
        RESPONSE_DIR,
        fname.replace(".txt", "_response.txt")
    )

    with open(prompt_path, "r", encoding="utf-8") as f:
        attack_features = f.read().strip()

    messages = build_messages(attack_features)

    try:
        print(f"\nQuerying {MISTRAL_MODEL} for: {fname}")

        llm_start = perf_counter()

        response_text = mistral_chat_complete(
            messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=TEMPERATURE
        )

        llm_end = perf_counter()

        # Do NOT save response, to avoid overwriting existing answers.
        # with open(response_path, "w", encoding="utf-8") as f:
        #     f.write(response_text)

        runtime_rows.append({
            "prompt_file": fname,
            "response_file": os.path.basename(response_path),
            "llm_generation_runtime_s": llm_end - llm_start
        })

        print(f"Generated response for {fname} but did not save it.")
        time.sleep(SLEEP_BETWEEN_CALLS)

    except Exception as e:
        print(f"[!] Failed on {fname}: {e}")
        continue

# ============ Runtime summary ============
runtime_df = pd.DataFrame(runtime_rows)

print("\n=== Runtime: Unconstrained LLM Mitigation Generation ===")

if len(runtime_df) == 0:
    print("No LLM calls were completed.")
else:
    total_runtime = runtime_df["llm_generation_runtime_s"].sum()
    avg_runtime = runtime_df["llm_generation_runtime_s"].mean()

    print(f"Generated responses: {len(runtime_df)}")
    print(f"Total runtime: {total_runtime:.6f} s")
    print(f"Average runtime per alert: {avg_runtime:.6f} s")
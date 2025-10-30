import os
import time
import json
import math
import random
from typing import List, Dict

# ============ CONFIG (Mistral API) ============
# Set your key as an env var: export MISTRAL_API_KEY=xxxx
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not set in environment")


MISTRAL_MODEL = "open-mistral-7b"

# Experiment/config
SLEEP_BETWEEN_CALLS = 1
USE_EXAMPLES = False
PROMPT_DIR = "prompts_exp"
BASE_OUT = "llm_mitigation_exp"
RESPONSE_DIR = os.path.join(BASE_OUT)
os.makedirs(RESPONSE_DIR, exist_ok=True)

DEFAULT_MAX_TOKENS = 1024
MIN_MAX_TOKENS = 64
TEMPERATURE = 0.0  # deterministic

# ============ Load context & (optional) examples ============
context_file = "context_mit.txt"
with open(context_file, "r") as f:
    initial_context = f.read().strip()


# ============ Mistral client ============
# pip install mistralai
from mistralai import Mistral

client = Mistral(api_key=MISTRAL_API_KEY)

# ============ Messaging ============
def build_messages(attack_features: str):
    msgs = [
        {"role": "system",
         "content": "You are a cybersecurity analyst specializing in wireless network attacks."},
        {"role": "user",
         "content": f"{initial_context}\n\nMost anomalous feature–value pairs:\n{attack_features}\n\n"
                    "Instructions:\n"
                    "- Summarize the likely cause in one paragraph.\n"
                    "- Provide a prioritized, actionable mitigation checklist (short bullets).\n"
                    "- Restrict to 5G NAS/RRC and core/RAN configuration where applicable.\n"
                    "- Be specific (timers, counters, rules, KPIs). State assumptions if evidence is insufficient."}
    ]
    return msgs

# ============ API call with retry/backoff ============
def mistral_chat_complete(messages: List[Dict[str, str]],
                          max_tokens: int = DEFAULT_MAX_TOKENS,
                          temperature: float = TEMPERATURE,
                          max_retries: int = 6) -> str:
    """
    Calls Mistral chat API with exponential backoff on 429/5xx.
    Returns assistant text.
    """
    # Cap tokens at a floor to avoid pathological retries
    max_tokens = max(MIN_MAX_TOKENS, max_tokens)

    for attempt in range(max_retries):
        try:
            resp = client.chat.complete(
                model=MISTRAL_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # You can add safe_prompt=True if your org enforces it.
            )
            # SDK returns: resp.choices[0].message.content
            if not resp or not resp.choices:
                raise RuntimeError("Empty response from Mistral API")
            text = resp.choices[0].message.content or ""
            return text.strip()
        except Exception as e:
            # Inspectable error text
            emsg = str(e).lower()

            # Heuristic: retry on rate limiting or server errors
            retryable = any(x in emsg for x in [
                "rate limit", "429", "timeout", "timed out", "service unavailable", "503", "bad gateway", "502"
            ])

            if attempt < max_retries - 1 and retryable:
                # exponential backoff with jitter
                backoff = (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(backoff)
                continue
            # If not retryable or out of retries, surface it
            raise

# ============ Runner ============
for fname in sorted(os.listdir(PROMPT_DIR)):
    if not fname.endswith(".txt"):
        continue

    prompt_path = os.path.join(PROMPT_DIR, fname)
    response_path = os.path.join(RESPONSE_DIR, fname.replace(".txt", "_response.txt"))

    with open(prompt_path) as f:
        attack_features = f.read().strip()

    messages = build_messages(attack_features)

    try:
        print(f"\nQuerying {MISTRAL_MODEL} for: {fname}")
        response_text = mistral_chat_complete(messages, max_tokens=DEFAULT_MAX_TOKENS, temperature=TEMPERATURE)
        with open(response_path, "w") as f:
            f.write(response_text)
        print(f"Saved response to {response_path}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    except Exception as e:
        print(f"[!] Failed on {fname}: {e}")
        continue

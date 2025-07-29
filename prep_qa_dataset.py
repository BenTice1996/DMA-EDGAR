# === prep_qa_dataset.py ===
# Downloads EDGAR contracts, normalizes them, and builds a QA dataset from human-labeled clauses

import os
import re
import json
import requests
import pandas as pd
import unicodedata
from bs4 import BeautifulSoup
from tqdm import tqdm

# === CONFIGURATION ===
CSV_PATH = "coded_contracts_with_ids.csv"
RAW_FOLDER = "raw_contracts"
TXT_FOLDER = "normalized_contracts"
OUTPUT_JSON = "qa_dataset.jsonl"
LOG_ERRORS_CSV = "unmatched_clauses.csv"

CLAUSE_COLUMNS = {
    "arb_text": "What is the arbitration clause?",
    "cof_text": "What is the choice of forum clause?",
    "col_text": "What is the choice of law clause?"
}

# === Ensure folders exist ===
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(TXT_FOLDER, exist_ok=True)

# === Normalize function ===
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()

# === Download and convert contract to normalized .txt ===
def fetch_and_normalize_contract(url, contract_id, log_list):
    basename = contract_id.replace("..", "__") + ".txt"
    raw_path = os.path.join(RAW_FOLDER, basename)
    txt_path = os.path.join(TXT_FOLDER, basename)

    # Skip if already processed
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    try:
        # Download
        if not os.path.exists(raw_path):
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(r.text)

        # Convert HTML to plain text
        with open(raw_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        raw_text = soup.get_text(separator=" ", strip=True)
        norm_text = normalize(raw_text)

        # Save normalized
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(norm_text)

        return norm_text

    except Exception as e:
        log_list.append({
            "contract_id": contract_id,
            "error_type": "fetch_or_parse_error",
            "message": str(e),
            "url": url,
            "clause_type": None,
            "clause_text": None
        })
        return None

# === Load and process CSV ===
df = pd.read_csv(CSV_PATH)
examples = []
error_log = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    contract_id = row["filename"].replace("/", "..").replace(":", "..")
    url = row.get("url")

    if not isinstance(url, str) or not url.startswith("http"):
        error_log.append({
            "contract_id": contract_id,
            "error_type": "missing_url",
            "message": "URL missing or malformed",
            "url": url,
            "clause_type": None,
            "clause_text": None
        })
        continue

    context = fetch_and_normalize_contract(url, contract_id, error_log)
    if not context:
        continue

    for col, question in CLAUSE_COLUMNS.items():
        clause_text = str(row.get(col, "")).strip()
        if clause_text and clause_text.lower() != "nan" and clause_text not in ["", "N/A"]:
            clause = normalize(clause_text)
            answer_start = context.find(clause)

            if answer_start == -1:
                error_log.append({
                    "contract_id": row["contract_id"],
                    "error_type": "clause_not_found",
                    "message": "Clause text not found in normalized contract",
                    "url": url,
                    "clause_type": col,
                    "clause_text": clause_text[:200]
                })
                continue

            qa_entry = {
                "contract_id": row["contract_id"],
                "context": context,
                "question": question,
                "answers": {
                    "text": [clause],
                    "answer_start": [answer_start]
                },
                "id": f"{row['contract_id']}_{col}",
                "filename": contract_id,
                "clause_type": col
            }
            examples.append(qa_entry)

# === Save QA dataset ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"\n✅ Saved {len(examples)} QA examples to {OUTPUT_JSON}")

# === Save unmatched/error logs ===
if error_log:
    pd.DataFrame(error_log).to_csv(LOG_ERRORS_CSV, index=False)
    print(f"⚠️ Logged {len(error_log)} errors to {LOG_ERRORS_CSV}")

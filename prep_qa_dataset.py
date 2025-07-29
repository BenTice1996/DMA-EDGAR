# === prep_qa_dataset.py ===
# Converts human-labeled clauses + GitHub-hosted HTML contracts into QA training format (normalized version)

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import re
import unicodedata

# === CONFIGURATION ===
CSV_PATH = "coded_contracts_with_ids.csv"
OUTPUT_JSON = "qa_dataset.jsonl"
LOG_UNMATCHED = "unmatched_clauses.log"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/padelson/dma_corpus/main/contracts/"

CLAUSE_COLUMNS = {
    "arb_text": "What is the arbitration clause?",
    "cof_text": "What is the choice of forum clause?",
    "col_text": "What is the choice of law clause?"
}

# === Normalization function ===
def normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text.strip()

# === Load data ===
df = pd.read_csv(CSV_PATH)
examples = []
unmatched = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    file_name = row["filename"]
    contract_url = GITHUB_RAW_BASE + file_name

    try:
        html = requests.get(contract_url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        context_raw = soup.get_text(separator=" ", strip=True)
        context = normalize(context_raw)
    except Exception as e:
        print(f"❌ Failed to fetch/parse {file_name}: {e}")
        continue

    for col, question in CLAUSE_COLUMNS.items():
        clause_text = str(row.get(col, "")).strip()
        if clause_text and clause_text.lower() != "nan" and clause_text not in ["", "N/A"]:
            clause = normalize(clause_text)
            answer_start = context.find(clause)

            if answer_start == -1:
                print(f"⚠️ Clause text not found in contract: {file_name} → {col}")
                unmatched.append({
                    "contract_id": row["contract_id"],
                    "filename": file_name,
                    "clause_type": col,
                    "clause_text": clause_text[:200]
                })
                continue

            qa_entry = {
                "context": context,
                "question": question,
                "answers": {
                    "text": [clause],
                    "answer_start": [answer_start]
                },
                "id": f"{row['contract_id']}_{col}",
                "filename": file_name,
                "clause_type": col
            }
            examples.append(qa_entry)

# === Save QA dataset ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"\n✅ Saved {len(examples)} QA examples to {OUTPUT_JSON}")

# === Save unmatched logs ===
if unmatched:
    with open(LOG_UNMATCHED, "w", encoding="utf-8") as f:
        for miss in unmatched:
            f.write(json.dumps(miss) + "\n")
    print(f"⚠️ Logged {len(unmatched)} unmatched clauses to {LOG_UNMATCHED}")

# === prep_qa_dataset.py ===
# Converts human-labeled clauses + GitHub-hosted HTML contracts into QA training format

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import json

# === CONFIGURATION ===
CSV_PATH = "coded_contracts_with_ids.csv"
OUTPUT_JSON = "qa_dataset.jsonl"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/padelson/dma_corpus/main/contracts/"
CLAUSE_COLUMNS = {
    "arb_text": "What is the arbitration clause?",
    "cof_text": "What is the choice of forum clause?",
    "col_text": "What is the choice of law clause?"
}

# === LOAD CLAUSE MAPPINGS ===
df = pd.read_csv(CSV_PATH)
examples = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    file_name = row["filename"]
    contract_url = GITHUB_RAW_BASE + file_name

    try:
        html = requests.get(contract_url).text
        soup = BeautifulSoup(html, "html.parser")
        context = soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"❌ Failed to fetch/parse {file_name}: {e}")
        continue

    for col, question in CLAUSE_COLUMNS.items():
        clause_text = str(row.get(col, "")).strip()
        if clause_text and clause_text.lower() != "nan" and clause_text not in ["", "N/A"]:
            # Try to find clause text offset in context
            answer_start = context.lower().find(clause_text.lower())
            if answer_start == -1:
                print(f"⚠️ Clause text not found in contract: {file_name} → {col}")
                continue

            qa_entry = {
                "context": context,
                "question": question,
                "answers": {
                    "text": [clause_text],
                    "answer_start": [answer_start]
                },
                "id": f"{row['contract_id']}_{col}",
                "filename": file_name,
                "clause_type": col
            }
            examples.append(qa_entry)

# === SAVE ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"✅ Saved {len(examples)} QA examples to {OUTPUT_JSON}")

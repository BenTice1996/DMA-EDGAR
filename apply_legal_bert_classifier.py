# === apply_legal_bert_classifier.py ===

import os
import re
import torch
import pandas as pd
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# === CONFIGURATION ===
UNCODED_DATA_CSV = "data_to_code_with_ids.csv"
UNCODED_CONTRACTS_DIR = "./uncoded_contracts/"
MODEL_DIR = "./legal_bert_clause_model"
OUTPUT_CSV = "extracted_clauses_wide.csv"
LABELS = ["none", "arbitration", "choice_of_forum", "choice_of_law", "equitable_carveout"]
MAX_LENGTH = 512
CHUNK_STRIDE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL AND TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

label_encoder = LabelEncoder().fit(LABELS)
nlp = spacy.load("en_core_web_sm")

# === HELPERS ===
def load_contract_text(contract_id):
    filepath = os.path.join(UNCODED_CONTRACTS_DIR, f"{contract_id}.txt")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")
        return ""

def chunk_text(text, max_tokens=MAX_LENGTH, stride=CHUNK_STRIDE):
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_tokens]
        if not chunk:
            continue
        decoded = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(decoded)
    return chunks

def extract_state_from_clause(clause):
    doc = nlp(clause)
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC"}:
            return ent.text
    return None

# === LOAD UNCODED METADATA ===
uncoded_df = pd.read_csv(UNCODED_DATA_CSV)
results = []

print(f"🔍 Applying model to {len(uncoded_df)} uncoded contracts...")

for _, row in tqdm(uncoded_df.iterrows(), total=len(uncoded_df)):
    contract_id = row["contract_id"]
    text = load_contract_text(contract_id)
    if not text.strip():
        continue

    chunks = chunk_text(text)
    clause_outputs = {
        "arbitration": None,
        "choice_of_forum": None,
        "forum_state": None,
        "choice_of_law": None,
        "law_state": None,
        "equitable_carveout": None
    }

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_label = label_encoder.inverse_transform([logits.argmax(dim=-1).item()])[0]

        if pred_label != "none" and clause_outputs[pred_label] is None:
            clause_outputs[pred_label] = chunk
            if pred_label == "choice_of_forum":
                clause_outputs["forum_state"] = extract_state_from_clause(chunk)
            elif pred_label == "choice_of_law":
                clause_outputs["law_state"] = extract_state_from_clause(chunk)

    result_row = row.to_dict()
    result_row.update(clause_outputs)
    results.append(result_row)

# === SAVE RESULTS ===
output_df = pd.DataFrame(results)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved extracted clauses to: {OUTPUT_CSV}")

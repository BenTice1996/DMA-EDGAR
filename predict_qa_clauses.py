# === predict_qa_clauses.py ===
# Applies fine-tuned Legal-BERT QA model to extract clauses from uncoded contracts (local TXT files)

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm
import torch.nn.functional as F

# === CONFIG ===
MODEL_PATH = "legalbert_qa_model"  # or "legalbert_qa_model_resume"
CONTRACT_FOLDER = "uncoded_contracts"  # Local directory with TXT files
OUTPUT_CSV = "predicted_clauses.csv"
CLAUSE_QUERIES = {
    "arbitration": "What is the arbitration clause?",
    "choice_of_forum": "What is the choice of forum clause?",
    "choice_of_law": "What is the choice of law clause?",
    "equitable_carveout": "What is the equitable carveout clause?"
}
MAX_LENGTH = 512

# === LOAD MODEL ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
model.eval()

def extract_text_from_txt(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return ""

# === PREDICT CLAUSES ===
txt_files = [f for f in os.listdir(CONTRACT_FOLDER) if f.endswith(".txt")]
predictions = []

for filename in tqdm(txt_files):
    filepath = os.path.join(CONTRACT_FOLDER, filename)
    context = extract_text_from_txt(filepath)
    if not context:
        continue

    row_result = {"filename": filename}

    for clause_type, question in CLAUSE_QUERIES.items():
        inputs = tokenizer(question, context, return_tensors="pt", truncation="only_second", max_length=MAX_LENGTH)
        with torch.no_grad():
            outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)
        tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
        answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()

        # Compute softmax probabilities
        start_prob = F.softmax(start_logits, dim=1)[0, start_idx].item()
        end_prob = F.softmax(end_logits, dim=1)[0, end_idx].item()
        confidence = round((start_prob * end_prob) ** 0.5, 4)  # geometric mean

        row_result[clause_type] = answer
        row_result[f"{clause_type}_confidence"] = confidence

    predictions.append(row_result)

# === SAVE OUTPUT ===
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved predictions for {len(pred_df)} contracts to {OUTPUT_CSV}")

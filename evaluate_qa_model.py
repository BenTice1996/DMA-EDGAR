# === evaluate_qa_model.py ===
# Compares model-extracted clause spans with human-annotated answers

import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import Dataset
from difflib import SequenceMatcher
from tqdm import tqdm

# === CONFIG ===
MODEL_DIR = "legalbert_qa_model"
DATA_PATH = "qa_dataset.jsonl"
MAX_LENGTH = 512
SIMILARITY_THRESHOLD = 0.9  # for fuzzy match success

# === LOAD MODEL ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)
model.eval()

# === LOAD QA DATA ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

results = []


def fuzzy_match(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


for item in tqdm(data):
    context = item["context"]
    question = item["question"]
    true_answer = item["answers"]["text"][0]

    encoding = tokenizer(question, context, return_tensors="pt", truncation="only_second", max_length=MAX_LENGTH)
    with torch.no_grad():
        outputs = model(**encoding)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)

    tokens = encoding["input_ids"][0][start_index:end_index + 1]
    predicted_answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    similarity = fuzzy_match(predicted_answer, true_answer)

    results.append({
        "id": item["id"],
        "question": question,
        "true_answer": true_answer,
        "predicted_answer": predicted_answer,
        "similarity": similarity,
        "correct": similarity >= SIMILARITY_THRESHOLD,
        "filename": item.get("filename"),
        "clause_type": item.get("clause_type")
    })

# === SAVE AND REPORT ===
results_df = pd.DataFrame(results)
results_df.to_csv("qa_evaluation_results.csv", index=False)

print("\n✅ Evaluation complete.")
print("📊 Match accuracy:")
print(results_df.groupby("clause_type")["correct"].mean().round(3))

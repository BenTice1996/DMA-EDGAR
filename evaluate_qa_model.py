import json
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from tqdm import tqdm
from difflib import SequenceMatcher

# === CONFIG ===
MODEL_DIR = "legalbert_qa_model/checkpoint-4404"
DATA_PATH = "qa_dataset.jsonl"
MAX_LENGTH = 512
DOC_STRIDE = 128
SIMILARITY_THRESHOLD = 0.8  # relaxed threshold

# === LOAD MODEL ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR)
model.eval()

# === LOAD QA DATA + DETERMINE MAX ANSWER LENGTH ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

max_answer_len = max(len(item["answers"]["text"][0].split()) for item in data)
MAX_ANSWER_LENGTH = max_answer_len + 10  # slight buffer

print(f"📏 Max answer length set to: {MAX_ANSWER_LENGTH} tokens")

# === Helper: Fuzzy Match ===
def fuzzy_match(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# === Evaluation Loop ===
results = []

for item in tqdm(data):
    context = item["context"]
    question = item["question"]
    true_answer = item["answers"]["text"][0]
    clause_type = item.get("clause_type")
    file_id = item.get("filename")
    sample_id = item.get("id")

    # Tokenize with sliding window
    encodings = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    best_score = float("-inf")
    best_answer = ""
    best_confidence = 0.0

    for i in range(len(encodings["input_ids"])):
        input_ids = encodings["input_ids"][i].unsqueeze(0)
        attention_mask = encodings["attention_mask"][i].unsqueeze(0)
        offset_mapping = encodings["offset_mapping"][i]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

        for start_idx in range(len(start_logits)):
            for end_idx in range(start_idx, min(start_idx + MAX_ANSWER_LENGTH, len(end_logits))):
                if offset_mapping[start_idx] is None or offset_mapping[end_idx] is None:
                    continue
                score = start_logits[start_idx] + end_logits[end_idx]
                if score > best_score:
                    start_char = offset_mapping[start_idx][0]
                    end_char = offset_mapping[end_idx][1]
                    best_answer = context[start_char:end_char]
                    best_score = score

                    # Softmax-based confidence
                    span_scores = start_logits + end_logits
                    span_probs = F.softmax(span_scores, dim=0)
                    best_confidence = span_probs[start_idx].item() * span_probs[end_idx].item()

    best_answer = best_answer.strip()
    similarity = fuzzy_match(best_answer, true_answer)

    results.append({
        "id": sample_id,
        "question": question,
        "true_answer": true_answer,
        "predicted_answer": best_answer,
        "similarity": similarity,
        "correct": similarity >= SIMILARITY_THRESHOLD,
        "confidence": round(best_confidence, 4),
        "filename": file_id,
        "clause_type": clause_type
    })

# === SAVE RESULTS ===
results_df = pd.DataFrame(results)
results_df.to_csv("qa_evaluation_results.csv", index=False)

print("\n✅ Evaluation complete.")
print("📊 Match accuracy by clause type:")
print(results_df.groupby("clause_type")["correct"].mean().round(3))

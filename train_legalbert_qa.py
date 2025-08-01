# === train_legalbert_qa.py ===
# Fine-tunes Legal-BERT on clause extraction using question answering (span prediction)

import json
import torch
from datasets import load_dataset, Dataset
import transformers
from transformers import TrainingArguments
from transformers import AutoModelForQuestionAnswering
from transformers import Trainer
from transformers import default_data_collator
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import evaluate

# === CONFIG ===
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DATA_PATH = "qa_dataset.jsonl"
OUTPUT_DIR = "C:/Users/Ben Tice/PycharmProjects/M&As_EDGAR_Scrape/legalbert_qa_model"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3

# === LOAD DATA ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    tokenized = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        padding="max_length",
        return_offsets_mapping=True
    )

    # Map answer span to token positions
    start_char = example["answers"]["answer_start"][0]
    end_char = start_char + len(example["answers"]["text"][0])
    offsets = tokenized["offset_mapping"]

    sequence_ids = tokenized.sequence_ids()
    context_start = sequence_ids.index(1)
    context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

    token_start = token_end = 0
    for idx in range(context_start, context_end):
        if offsets[idx] is None:
            continue
        if offsets[idx][0] <= start_char < offsets[idx][1]:
            token_start = idx
        if offsets[idx][0] < end_char <= offsets[idx][1]:
            token_end = idx
            break

    tokenized["start_positions"] = token_start
    tokenized["end_positions"] = token_end
    tokenized.pop("offset_mapping")
    return tokenized

processed_dataset = dataset.map(preprocess)

# === MODEL ===
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# === TRAINING ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    logging_dir="./logs_qa",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    push_to_hub=False
)

squad_metric = evaluate.load("squad")

def compute_metrics(p):
    return squad_metric.compute(
        predictions=[{
            "id": ex["id"],
            "prediction_text": pred
        } for ex, pred in zip(p.label_ids, p.predictions["text"])],
        references=[{
            "id": ex["id"],
            "answers": ex["answers"]
        } for ex in p.label_ids]
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    eval_dataset=processed_dataset.shuffle(seed=42).select(range(100)),  # small held-out subset
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=None  # Optional: squad metrics require post-processing
)

# === TRAIN ===
print("\n🚀 Training Legal-BERT for clause extraction...")
trainer.train()
print("\n✅ Model training complete. Saved to:", OUTPUT_DIR)

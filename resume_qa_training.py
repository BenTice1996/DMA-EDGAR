# === resume_qa_training.py ===
# Resumes fine-tuning a Legal-BERT QA model from a saved checkpoint

import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)

# === CONFIG ===
CHECKPOINT_DIR = "legalbert_qa_model"  # change if resuming from subfolder
DATA_PATH = "qa_dataset.jsonl"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 2  # Additional epochs
RESUME_OUTPUT_DIR = "legalbert_qa_model_resume"

# === LOAD DATA ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

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
model = AutoModelForQuestionAnswering.from_pretrained(CHECKPOINT_DIR)

# === TRAINING ===
training_args = TrainingArguments(
    output_dir=RESUME_OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    logging_dir="./logs_qa_resume",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    resume_from_checkpoint=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    eval_dataset=processed_dataset.shuffle(seed=42).select(range(100)),
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=None
)

# === CONTINUE TRAINING ===
print("\n🔁 Resuming QA model training...")
trainer.train()
print("\n✅ Training resumed and model saved to:", RESUME_OUTPUT_DIR)

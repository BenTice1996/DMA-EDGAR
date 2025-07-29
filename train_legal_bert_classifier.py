# === train_legal_bert_classifier.py ===

import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# === CONFIGURATION ===
TRAINING_DATA_CSV = "coded_contracts_with_ids.csv"
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
OUTPUT_DIR = "./legal_bert_clause_model"
PARTIAL_OUTPUT_DIR = "./legal_bert_clause_model_partial"
LABELS = ["none", "arbitration", "choice_of_forum", "choice_of_law", "equitable_carveout"]
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD DATA ===
df = pd.read_csv(TRAINING_DATA_CSV)
clause_records = []

# Extract one row per clause
for _, row in df.iterrows():
    for col, label in [
        ("arb_clause", "arbitration"),
        ("cof_clause", "choice_of_forum"),
        ("col_clause", "choice_of_law"),
        ("equitable_carveout", "equitable_carveout"),
    ]:
        clause = str(row.get(col, "")).strip()
        if clause and clause.lower() != "nan":
            clause_records.append({
                "contract_id": row["contract_id"],
                "text": clause,
                "label": label
            })

df_clean = pd.DataFrame(clause_records)
print(f"✅ Loaded {len(df_clean)} labeled clauses from {len(df['contract_id'].unique())} contracts.")

# === LABEL ENCODING ===
label_encoder = LabelEncoder().fit(LABELS)
df_clean["label_id"] = label_encoder.transform(df_clean["label"])

# === TRAIN/VAL SPLIT ===
df_train, df_val = train_test_split(
    df_clean,
    test_size=0.2,
    stratify=df_clean["label"],
    random_state=42
)

# === TOKENIZATION ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

train_dataset = Dataset.from_pandas(df_train[["text", "label_id"]])
val_dataset = Dataset.from_pandas(df_val[["text", "label_id"]])

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

train_dataset = train_dataset.rename_column("label_id", "labels")
val_dataset = val_dataset.rename_column("label_id", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# === MODEL + TRAINING ===
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS)
).to(device)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_dir="./logs",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# === TRAIN ===
print("🚀 Training Legal-BERT...")
trainer.train()

# === EVALUATE ===
print("📊 Final evaluation on validation set:")
preds = trainer.predict(val_dataset)
y_true = preds.label_ids
y_pred = preds.predictions.argmax(axis=1)

report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
print(report)

# === ALWAYS SAVE INTERMEDIATE MODEL ===
trainer.save_model(PARTIAL_OUTPUT_DIR)
print(f"💾 Partial model saved at {PARTIAL_OUTPUT_DIR}")

# === CHECK IF F1 THRESHOLD MET ===
report_dict = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)
ok = True
for label in ["arbitration", "choice_of_forum", "choice_of_law", "equitable_carveout"]:
    acc = report_dict[label]["f1-score"]
    if acc < 0.95:
        ok = False
        print(f"❌ {label} F1 below 95%: {acc:.2%}")
    else:
        print(f"✅ {label} F1: {acc:.2%}")

# === SAVE FINAL MODEL IF THRESHOLD MET ===
if ok:
    print("✅ All clause types meet accuracy threshold. Saving final model...")
    trainer.save_model(OUTPUT_DIR)
else:
    print("⚠️ Model not saved as final — accuracy threshold not met.")

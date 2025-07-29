import pandas as pd

# === CONFIG ===
TRAINING_DATA_CSV = "coded_contracts.csv"
UNCODED_DATA_CSV = "data_to_code.csv"
UPDATED_TRAINING_CSV = "coded_contracts_with_ids.csv"
UPDATED_UNCODED_CSV = "data_to_code_with_ids.csv"

# === LOAD BOTH DATASETS ===
df_train = pd.read_csv(TRAINING_DATA_CSV)
df_uncoded = pd.read_csv(UNCODED_DATA_CSV)

# === GENERATE UNIQUE CONTRACT IDS ===
total_rows = len(df_train) + len(df_uncoded)
contract_ids = [f"C{i:06d}" for i in range(1, total_rows + 1)]

# === ASSIGN IDS TO EACH FILE ===
df_train = df_train.copy()
df_train["contract_id"] = contract_ids[:len(df_train)]

df_uncoded = df_uncoded.copy()
df_uncoded["contract_id"] = contract_ids[len(df_train):]

# === SAVE UPDATED FILES ===
df_train.to_csv(UPDATED_TRAINING_CSV, index=False)
df_uncoded.to_csv(UPDATED_UNCODED_CSV, index=False)

print(f"✅ Saved: {UPDATED_TRAINING_CSV} ({len(df_train)} rows)")
print(f"✅ Saved: {UPDATED_UNCODED_CSV} ({len(df_uncoded)} rows)")

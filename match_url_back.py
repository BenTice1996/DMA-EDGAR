import pandas as pd

# Load old and new CSVs
old = pd.read_csv("coded_contracts_with_ids.csv")  # has 'url'
new = pd.read_csv("coded_contracts_post_with_ids.csv")  # missing 'url'

# Merge URL back using 'FactSetID' (or use 'filename' if needed)
merged = pd.merge(new, old[["FactSetID", "url"]], on="FactSetID", how="left")

# Save corrected file
merged.to_csv("coded_contracts_post_with_urls.csv", index=False)
print("✅ Saved corrected CSV with URLs.")

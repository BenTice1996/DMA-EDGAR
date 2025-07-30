import pandas as pd

df = pd.read_csv("coded_contracts_post_with_urls.csv")

print("✅ Total rows:", len(df))
print("✅ Unique FactSetIDs:", df['FactSetID'].nunique())
print("✅ Duplicate rows:", df.duplicated().sum())
print("✅ Duplicate FactSetIDs:", df['FactSetID'].duplicated().sum())

# Drop rows with missing URL
df = df[df['url'].notna()]

# Deduplicate by FactSetID (keep the first with a URL)
df = df.drop_duplicates(subset="FactSetID", keep="first")

# Sanity check
print("✅ Final rows:", len(df))
print("✅ Unique FactSetIDs:", df['FactSetID'].nunique())

# Save result
df.to_csv("coded_contracts_post_with_urls_deduped.csv", index=False)
print("✅ Saved deduplicated CSV with only unique FactSetIDs and valid URLs.")
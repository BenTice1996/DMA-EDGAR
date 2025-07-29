import os
import pandas as pd
import requests
import shutil
from tqdm import tqdm
import html2text
from urllib.parse import urlparse

# === Config ===
CONTRACTS_DIR = "Contracts"
FINAL_DIR = "FinalContracts"
CSV_PATH = "data_to_code.csv"

os.makedirs(FINAL_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(CSV_PATH)
urls = df['url'].dropna().tolist()

# === Normalize URLs to Base IDs ===
def extract_base_id(full_url):
    try:
        path = urlparse(full_url).path  # Strip domain, leave just the path
        parts = path.strip('/').split('/')
        if len(parts) < 6:
            return None
        cik = parts[3]
        acc1 = parts[4]
        acc2 = parts[5].split('.')[0]  # Remove .htm, .html, .txt, etc.
        return f"{cik}..{acc1}..{acc2}"
    except Exception:
        return None

# === HTML to Text Converter ===
def convert_html_to_text(html):
    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.body_width = 0
    return converter.handle(html)

# === Step 1: Try Local Match, Step 2: Download and Convert ===
matched, downloaded, failed = [], [], []

for url in tqdm(urls, desc="Processing contracts"):
    base_id = extract_base_id(url)
    if not base_id:
        failed.append(url)
        continue

    found_local = False
    for filename in os.listdir(CONTRACTS_DIR):
        if base_id in filename:
            src_path = os.path.join(CONTRACTS_DIR, filename)
            dest_path = os.path.join(FINAL_DIR, f"{base_id}.txt")
            try:
                # Copy and convert to plain text if needed
                with open(src_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if filename.lower().endswith(('.htm', '.html')):
                    content = convert_html_to_text(content)
                with open(dest_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(content)
                matched.append(base_id)
                found_local = True
            except Exception as e:
                failed.append(url)
            break

    if not found_local:
        try:
            full_url = f"https://www.sec.gov{url}"
            response = requests.get(full_url, headers={"User-Agent": "Ben Tice tice2@law.upenn.edu"}, timeout=10)
            if response.status_code == 200:
                text = convert_html_to_text(response.text)
                dest_path = os.path.join(FINAL_DIR, f"{base_id}.txt")
                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(text)
                downloaded.append(base_id)
            else:
                failed.append(url)
        except Exception:
            failed.append(url)

# === Logging Summary ===
print(f"\n✅ Local matches copied and converted: {len(matched)}")
print(f"🌐 Contracts downloaded and converted: {len(downloaded)}")
print(f"❌ Failed: {len(failed)}")

# Optional: Save failures to CSV
if failed:
    pd.DataFrame(failed, columns=["url"]).to_csv("failed_contracts.csv", index=False)

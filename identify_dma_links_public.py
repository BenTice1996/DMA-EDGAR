import os
import re
import time
import requests  # ✅ Add this line
import pandas as pd
from bs4 import BeautifulSoup
import html as html
import html2text
import datetime as dt
import random as rand
from collections import defaultdict
from tqdm import tqdm
import traceback

CONTRACTS_DIR = "C:/Users/Ben Tice/PycharmProjects/M&As_EDGAR_Scrape/Contracts"

WINDOW_START = 1 #days before fact_set announcement date to look for an SEC filing
WINDOW_END = 45 #days after fact_set announcement date to look for an SEC filing

matched = 0
unmatched = 0
errors = []

fact_set = pd.read_excel('small_deals.xlsx', sheet_name='Transactions_Results')
print(f"✅ Loaded {len(fact_set)} rows from Excel")  # 👈 ADD THIS LINE

# Rename columns to match expected input fields
fact_set.rename(columns={
    'target_cusip': 'Target Cusip_fs',
    'buyer_cusip': 'Acquirer Cusip_fs',
    'Target/Issuer Name': 'Target_fs',
    'Ultimate Parent - At Deal  (Target/Issuer)': 'Target Ultimate Parent (At Deal)_fs',
    'Buyer Name': 'Acquirer_fs',
    'Ultimate Parent - At Deal  (Buyer/Investor)': 'Acquirer Ultimate Parent (At Deal)_fs',
    'Announcement Date': 'Announcement Date_fs'
}, inplace=True)

# Create dummy completion date column
fact_set['Completion Date_fs'] = fact_set['Announcement Date_fs']

# ✅ THEN convert to list of dicts
fact_set_records = fact_set.to_dict('records')

common_words = []
with open('1000_common_words.txt') as f:
    lines = f.readlines()
    for line in lines:
        common_words.append(line.strip())

print("Reading in CIK-CUSIP csv") #Progress printout

#Construct to the cusip to cik. Based on data from https://github.com/leoliu0/cik-cusip-mapping/blob/master/cik-cusip-maps.csv
cusip_to_cik = pd.read_csv('cik-cusip-maps.csv')
cusip_to_cik['cik'] = cusip_to_cik['cik'].astype(str)
cusip_to_cik['cik'] = cusip_to_cik['cik'].apply(lambda x: x[0:x.find('.')])
cusip_to_cik = cusip_to_cik.drop_duplicates(subset = ['cusip8'])
cusip_to_cik = cusip_to_cik.set_index('cusip8')
cusip_to_cik = cusip_to_cik.to_dict('index')
for key, v in cusip_to_cik.items():
    cusip_to_cik[key] = v['cik']

print("Reading in contracts csv") #Progress printout

links = pd.read_csv("contracts_cleaned_parties.csv")
links['date.filed'] = links['date.filed'].apply(lambda x: pd.to_datetime(x))
links = links.sort_values(by = ['date.filed', 'Unnamed: 0'])
links['cik'] = links['cik'].apply(lambda x: str(x))
ciks =  links['cik'].unique()
links['exhibit_lead'] = links['exhibit'].apply(lambda x: str(x).split('.')[0] if pd.notna(x) else '')

def find_ciks_from_cusip(company_info):
    co_ciks = []
    if not company_info or 'cusips' not in company_info:
        return co_ciks
    for cusip in company_info['cusips']:
        cik = find_cik_from_cusip(cusip)
        if cik and cik in ciks:
            co_ciks.append(cik)
    return co_ciks

def clean_name(raw_name): #clean the names in the fact_set dataset
    if not isinstance(raw_name, str):
        return []
    i_names = raw_name.split(';') #delimiter for names
    names = []
    for i_name in i_names:
        name = i_name.lower()
        names.append(name)
        if '&' in name:
            names.append(name.replace('&', 'and'))
    final_names = []
    suffix = ['inc', 'corp', 'corporation', 'incorpated', 'limited', 'llc', 'llp', 'lmt', 'ltd', 'co', 'lp', 'partners']
    for name in names:
        name = name.replace('.', '').replace(',','').replace('A/S', '').replace('  ',' ')
        if name.count('(') > 0:
            indx_1 = name.find('(')
            indx_2 = name.find(')')
            if indx_2 > indx_1: 
                name = name[:indx_1] + name[indx_2+1:]
            else: #should never happen based on my data inspection
                name.replace('(', '').replace(')','')
        if name.count('/') > 0: #exclude the asset from company name
            name = name[:name.find('/')]
        if name == '-':
            continue
        final_names.append(name.strip())
        t_name = name.lower().strip().split(' ')
        t_name_end = t_name[-1]
        for s in suffix: #add name without various corporate suffixes to make matching a bit easier
            if t_name_end == s:
                final_names.append(' '.join(t_name[:-1]))
    return final_names

def clean_cusips(cusips):
    if not isinstance(cusips, str):
        return []
    cusips = cusips.split(';')  # ; delimited
    final_cusips = []
    for cusip in cusips:
        cusip = cusip.strip()
        if cusip == '-' or cusip == '':
            continue
        final_cusips.append(cusip)
    return final_cusips

def get_company_info(data, target=0):
    # Determine prefix based on target/acquirer
    start = "Target" if target == 1 else "Acquirer"

    company_info = {'cusips': [], 'names': []}

    # --- Try Primary CUSIP ---
    primary_cusip_key = f"{start} Cusip_fs"
    if primary_cusip_key in data and pd.notna(data[primary_cusip_key]):
        company_info['cusips'].extend(clean_cusips(data[primary_cusip_key]))
    else:
        # Log or handle missing primary CUSIP
        pass  # optional: print(f"⚠️ Missing {primary_cusip_key}")

    # --- Try Ultimate Parent CUSIP (if you ever add this) ---
    # This key does NOT exist in your data, so we skip or leave as placeholder
    # ultimate_cusip_key = f"{start} Ultimate Parent Cusip_fs"
    # if ultimate_cusip_key in data and pd.notna(data[ultimate_cusip_key]):
    #     company_info['cusips'].extend(clean_cusips(data[ultimate_cusip_key]))

    # --- Clean Entity Names ---
    name_keys = [
        f"{start}_fs",  # e.g., 'Acquirer_fs'
        f"{start} Ultimate Parent (At Deal)_fs"  # e.g., 'Acquirer Ultimate Parent (At Deal)_fs'
    ]
    for key in name_keys:
        if key in data and pd.notna(data[key]):
            company_info['names'].extend(clean_name(data[key]))

    return company_info

def find_cik_from_cusip(cusip):
    if not isinstance(cusip, str):
        return None
    for length in [8, 6]:
        try:
            return cusip_to_cik[cusip[:length]]
        except KeyError:
            continue
    return None

def create_name_dict(t_links):
    names_dict = t_links.drop_duplicates(subset = ['company.name'])
    names_dict = names_dict.set_index('company.name')
    names_dict = names_dict.to_dict('index')
    names_dict_final = {}
    for key, v in names_dict.items():
        names_dict_final[key.lower()] = v['cik']
    return names_dict_final

def levenshteinDistance(s1, s2): #consider making a fast fail if this runs slow
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def find_ciks_from_name(company_info, date):
    ciks = []
    window_start = date - dt.timedelta(days=WINDOW_START)
    window_end = date + dt.timedelta(days=WINDOW_END)
    t_links = links[links['date.filed'].apply(lambda x: x >= window_start and x<=window_end)] #only consider filings in window
    name_to_cik = create_name_dict(t_links)
    link_names = list(name_to_cik.keys())
    for name in company_info['names']:
        name = name.lower()
        min_d = 1000
        best_match = ''
        for link_name in link_names:
            d = levenshteinDistance(name, link_name)
            if d < min_d:
                best_match = link_name
                min_d = d
        #print (best_match)
        if min_d < max(len(name)/3, 3) and min_d < 11: 
            #similarity threshold for a match - must be less than max(1/3 length, 3) and no more than 11
            #arbitrarily tuned by hand for performance
            #-----ideally will run tests on how threshold can be optimized for false positives / false negatives
            ciks.append(name_to_cik[best_match])
    return ciks

def get_candidates(cik, date): #2000 has some odd filing formats - need a special case to deal with those
    #print('cik: '+ cik)
    t_links = links[links['cik'].apply(lambda x: str(x) == cik)]
    #print(len(t_links))
    window_start = date - dt.timedelta(days=WINDOW_START)
    window_end = date + dt.timedelta(days=WINDOW_END)
    t_links = t_links[t_links['date.filed'].apply(lambda x: x >= window_start and x<=window_end)]
    #print(len(t_links))
    cans= list(t_links['contract.link'])
    #print (cans)
    return cans

def find_dma_contract(acq_company_info, dates, target_company_info):
    acq_ciks = find_ciks_from_cusip(acq_company_info) or []
    tar_ciks = find_ciks_from_cusip(target_company_info) or []
    checked_links = []
    acq_ciks_name = 'GO'
    tar_ciks_name = 'GO'
    acq_use_name = 1
    tar_use_name = 1
    best_link = ''
    best_score = 0
    for date in dates: #test announcement date then completion date
        #print(date)
        for cik in acq_ciks: #test acquirer ciks
            cans = get_candidates(cik, date)
            for can in cans:
                if can in checked_links: 
                    continue
                checked_links.append(can)
                sc = is_ma_agreement(can, target_company_info['names'], acq_company_info['names'])
                if sc == 1: 
                    return can, 1
                elif sc > best_score:
                    best_link = can
                    best_score = sc
        for cik in tar_ciks: #test target ciks
            cans = get_candidates(cik, date)
            if cik in acq_ciks or cik in acq_ciks_name: #already tested cik
                continue
            for can in cans:
                if can in checked_links: 
                    continue
                checked_links.append(can)
                sc = is_ma_agreement(can, acq_company_info['names'], target_company_info['names'])
                if sc == 1: 
                    return can, 1
                elif sc > best_score:
                    best_link = can
                    best_score = sc
        if acq_ciks_name == 'GO':
            acq_ciks_name = find_ciks_from_name(acq_company_info, date) #no hit from cusip - go to name
        for cik in acq_ciks_name: #test acquier names
            if cik in acq_ciks: #already tested cik
                continue
            cans = get_candidates(cik, date)
            for can in cans:
                if can in checked_links: 
                    continue
                checked_links.append(can)
                sc = is_ma_agreement(can, target_company_info['names'], acq_company_info['names'])
                if sc == 1: 
                    return can, 1
                elif sc > best_score:
                    best_link = can
                    best_score = sc
        if tar_ciks_name == 'GO':
            tar_ciks_name = find_ciks_from_name(target_company_info, date) #no hit from cusip - go to name
        for cik in tar_ciks_name: #test target names
            if cik in tar_ciks or cik in acq_ciks or cik in acq_ciks_name: #already tested cik
                continue
            cans = get_candidates(cik, date)
            for can in cans:
                if can in checked_links: 
                    continue
                checked_links.append(can)
                sc = is_ma_agreement(can, acq_company_info['names'], target_company_info['names'])
                if sc == 1: return can, 1
                elif sc > best_score:
                    best_link = can
                    best_score = sc
    return best_link, best_score    

def dma_contract(data):
    acq_info = get_company_info(data)
    target_info = get_company_info(data, target = 1)
    dates_t = [data['Announcement Date_fs'], data['Completion Date_fs']] #sometimes need to check both dates
    dates_t = [d for d in dates_t if pd.notna(d)]
    if not dates_t:
        return '', 0  # skip if no usable dates
    if dates_t[1] == dates_t[0]: dates_t = [dates_t[0]] #same date - no difference
    dates = []
    for date in dates_t:
        try:
            dates.append(pd.to_datetime(date))
        except:
            pass
        if len(dates) > 1 and dates[1] - dates[0] < dt.timedelta(days=WINDOW_END):
            dates[1] = dates[0] + dt.timedelta(days=WINDOW_END) + dt.timedelta(days=WINDOW_START) #ensure no overlap in dates
    link, res = find_dma_contract(acq_info, dates, target_info) #acquier
    if res == 0: #failed
        return '', 0 #failure code
    return "https://www.sec.gov"+link, res

def name_match_score(text, names):
    best = 0
    for name in names:
        name = name.lower()
        split_name = name.split(' ')
        if name in text: 
            if any(c.isalpha() for c in name) and len(name) > 4 and name not in common_words:
                return 1 #exact match with sufficiently long name not in comon words
            elif name not in common_words: #exact match with a shorter name
                if best < .8: best = .8
            else: #exact match with a shorter, common name - not enough to report a real match
                best = .49
        if len(split_name) > 2:
            if ' '.join(split_name[:-1]) in text: #just last word missing
                best = .99
        if len(split_name) > 3: #just two words missing
            if ' '.join(split_name[:-2]) in text: 
                if best < .95: best = .95
        t = [split_name[0]+' ', split_name[0]+',', split_name[0]+'.', split_name[0]+'\n'] #check for first word match
        if any(t_name in text for t_name in t):
            if split_name[0] in common_words:
                continue
            if any(c.isalpha() for c in split_name[0]):
                if len(split_name[0]) > 3:
                    if best < .9: best = .9
                if best < .8: best = .8
            else:
                if len(split_name[0]) > 4:
                    if best < .8: best = .8
                else:
                    if best < .51: best = .51
    return best

def is_ma_agreement(contract_link, non_filer_names = [], filer_names= []): #run cell below first for relevant code
    text = extract_text(get_filename(contract_link)).lower()
    text = text.replace('\n\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    types = contract_type(text, types_dict)
    reported_type = types['type']
    name_match_filer = name_match_score(text[:1000], filer_names)
    name_match_non_filer = name_match_score (text[:1000], non_filer_names)
    if reported_type == 'ma':
        return max(name_match_non_filer, .51)
    elif reported_type == 'asset':
        return max(name_match_non_filer, .5)
    elif reported_type == 'equity':
        return max(name_match_non_filer, .4)
    else: #catch for catch-all phrases
        catch_alls = ['sale and purchase agreement', 'purchase and sale agreement', 
                      'purchase agreement', 'agreement of purchase and sale',
                     'agreement of sale and purchase', 'contribution agreement', 
                      'contribution, convetance, assumption, and simplification agreement',
                     "sale agreement", 'transaction agreement']
        start = text[:1000].replace('\n',' ').replace('   ', ' ').replace('  ',' ')
        if any(agreement in start for agreement in catch_alls):
            if name_match_non_filer > .5:
                return (name_match_non_filer + name_match_filer) / 2
    return 0

def get_filename(contract_link):
    ls = contract_link.split("/")
    fn = ls[4]+".."+ls[5]+".."+ls[6]
    return fn

def extract_text(fn):
    """Fetches and extracts text from a contract file. Downloads from SEC if not found locally."""
    filepath = os.path.join(CONTRACTS_DIR, fn)

    # If file is missing, try to download it
    if not os.path.exists(filepath):
        tqdm.write(f"🌐 Downloading missing file: {fn}")
        try:
            # Construct the URL
            parts = fn.split('..')
            if len(parts) != 3:
                raise Exception(f"Invalid contract filename format: {fn}")
            url = f"https://www.sec.gov/Archives/edgar/data/{parts[0]}/{parts[1]}/{parts[2]}"

            # Download the file from SEC
            headers = {"User-Agent": "Ben Tice tice2@law.upenn.edu"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Save it locally
            with open(filepath, 'wb') as f:
                f.write(response.content)

        except Exception as e:
            tqdm.write(f"❌ Failed to download contract: {fn} – {e}")
            raise

    # Extract and clean text
    try:
        with open(filepath, "rb") as f:
            htm = f.read().decode('utf8', errors='ignore')
            htm = re.sub('&#147;', '"', htm)
            htm = re.sub('&#148;', '"', htm)
            htm = re.sub('<B>', '', htm)
            htm = re.sub('</B>', '', htm)
            if len(htm) > 9000000:
                tqdm.write(f"⚠️ Skipping large file: {fn}")
                raise Exception('File too large')
            text = html2text.html2text(htm)
            document_text = text.replace('\\t', '')
            document_text = re.sub(r'\n.{0,10}\n.{0,10}\n.{0,10}\n.{0,10}\n*\* \* \*\n+(.*)\n+', ' ', document_text)
            document_text = re.sub(r'\n +\n', '\n\n', document_text)
            document_text = re.sub(r'  +', ' ', document_text)
            document_text = document_text.replace('&amp;', 'and')
            document_text = document_text.replace('&', 'and')
            document_text = document_text.replace("\\n", " ")
        return document_text
    except Exception as e:
        tqdm.write(f"❌ Failed to extract text from {fn}: {e}")
        return ""  # safer to fail gracefully

def contract_type(text, types_dict): #HOW TO DEAL WITH AMENDED AND RESTATED AGREEMENTS - RESTATED MEANS IT's A FULL CONTRACT
    ''' A Function that accepts a string (here, the first instance of agreement and the preceding words) and a dictionary
    with words that defines triggers to categorize the type of contract. Keys needs to be ma, license, legal, loan, employment and
    incentive'''
    # In my understanding, you look for the first occurence of a "format" word, and then pulls all text ahead of it.
    # This function then categorizes the contract based on the first occurrence of a keyword
    #This function also categorizes contracts as ammendments based on the text before the format word

    my_dict = {}
    the_keys = types_dict.keys()
    the_type = 'undefined'
    text = text.lower()
    text = re.sub(r"\n", ' ', text)
    text = re.sub("\t", ' ', text)
    text = re.sub(r"[/-]", ' ', text)
    text = re.sub(r"[^a-z ]+", '', text)
    text = re.sub(r'\s+', ' ', text)
    formats = ["agreement", "plan", "note", "policy", "guideline", "program", "contract"]

    occurrences = {}
    for word in formats:
        position = text.find(' '+word)
        if position == -1:
            continue
        occurrences[word] = position
    if len(occurrences) > 0:
        the_format = min(occurrences, key=occurrences.get)
    else: the_format = 'undefined'

    occurrences = {}
    for key in the_keys:
        word_list = types_dict[key]
        for word in word_list:
            position = text.find(' '+word)
            if position == -1:
                continue
            occurrences[word] = position
    if len(occurrences) > 0:
        the_word = min(occurrences, key=occurrences.get)
        pos = occurrences[the_word]
        relevant_text = text[: min(len(text), pos+50)]
    else:
        the_word = 'undefined'
        relevant_text = ""

    for k, v in types_dict.items():
        if the_word in v:
            the_type = k

    #print the_type
    my_dict['type'] = the_type
    if 'amendm' in relevant_text or 'amending' in relevant_text:
        my_dict['amendment'] = 1
    else:
        my_dict['amendment'] = 0
    if 'restat' in relevant_text:
        my_dict['restate'] = 1
    else:
        my_dict['restate'] = 0
    my_dict['type_hit_word'] = the_word
    my_dict['type_text'] = relevant_text
    my_dict['format'] = the_format
    return my_dict

# Agreement Types and their respective word triggers - taken from Julian Nyarko's old code

incentives = ("pension", 'stock unit', 'award', 'incentive', 'compensation','management stability', 'stock option', 'restricted stock', 'tax deferred savings','reimbursement', 'retention', 'separation allowance', 'retirement', 'bonus', 'dsu', 'medical plan', 'benefit', 'indemnification', 'health plan','executive plan', 'savings and investment', 'stock ownership', 'restoration plan', 'performance share', 'stock retainer', 'performance plan', 'management stockholders', 'indemnity', 'director stock', 'directors stock')

other = ('registration rights', 'omnibus', 'general conditions', 'share appreciation', 'limited liability company agreement')

employment = ('employer', 'employee', 'employment','severance', 'non competition', 'termination', 'management continuity', 'transition', 'appointment')

lease = ('lease', 'line access', 'sublease', 'tenant', 'landlord')

sales = ('distribution', 'repurchase')

ni = ('promissory', 'absldas')

equity_purchase = ('share purchase', 'stock purchase', 'securities purchase','share exchange', 'unit purchase agreement','membership interest purchase', 'membership interest exchange', 'membership interests purchase', 'membership interests exchange', "membership interest contribution", "membership interests contribution",'equity purchase', 'equity exchange', 'stock exchange agreement',"share sale", "share swap", "equity interests purchase", "interest purchase agreement")

asset_purchase = ('asset purchase', 'asset purchase agreement', 'asset sale')

loan = ('credit', 'loan', 'subordination',
       'borrow', 'lender', 'commitment')

ma = ( 'merger',  'arrangement agreement','acquisition agreement', 'amalgamation', 'combination')
#ma = ('change in control', 'change of control', 'share exchange', 'merger', 'separation and distribution', 'earnout', 'earn out')
#add cooperation? - going to miss e.g. take-two takeover agreement (first contract in csv)

jv = ('joint venture', 'point penture')

license = ('license', 'licensing')

legal = ('settlement', 'tolling', 'waiver')

types_dict = {}
types_dict['incentives'] = incentives
types_dict['employment'] = employment
types_dict['asset'] = asset_purchase
types_dict['equity'] = equity_purchase
types_dict['sales'] = sales
types_dict['loan'] = loan
types_dict['ma'] = ma
types_dict['license'] = license
types_dict['legal'] = legal
types_dict['lease'] = lease
types_dict['ni'] = ni
types_dict['jv'] = jv
types_dict['other'] = other

# === Load restart checkpoint if exists ===
restart_file = 'restart_index.txt'
restart_index = 0
if os.path.exists(restart_file):
    with open(restart_file, 'r') as f:
        try:
            restart_index = int(f.read().strip())
            print(f"🔁 Restarting from index {restart_index}")
        except ValueError:
            print("⚠️ restart_index.txt is invalid. Starting from index 0.")

# === Validate record count ===
print(f"🧮 Total records in fact_set_records: {len(fact_set_records)}")
EXPECTED_ROWS = 165938
if len(fact_set_records) != EXPECTED_ROWS:
    raise ValueError(f"❌ Row count mismatch! Expected {EXPECTED_ROWS} records but got {len(fact_set_records)}.\n"
                     "💡 Check if you're reading the wrong file or overwrote part of the data.")

for indx in tqdm(range(restart_index, len(fact_set_records)), desc="Processing deals", initial=restart_index, total=len(fact_set_records)):
    r = fact_set_records[indx]

    try:
        link, sc = dma_contract(r)
        r['url'] = link
        r['sc'] = sc

        if sc == 0:
            unmatched += 1
        else:
            matched += 1

    except Exception as e:
        r['url'] = ''
        r['sc'] = 0
        errors.append(r)
        tqdm.write(f"❌ Runtime error on record {indx}: {e}")
        traceback.print_exc()

    # ✅ Write restart checkpoint after each iteration
    with open('restart_index.txt', 'w') as f:
        f.write(str(indx))

    # Periodic checkpoint update every 500 deals
    if indx % 500 == 0 and indx > 0:
        tqdm.write(f"📊 Progress at {indx} — ✅ Matches: {matched} | ⚠️ Unmatched: {unmatched} | ❌ Errors: {len(errors)}")

        # 🔐 Write to a checkpoint file instead of overwriting the master
        pd.DataFrame(fact_set_records).to_csv('matched_records_checkpoint.csv', index=False)
        pd.DataFrame(errors).to_csv('errors.csv', index=False)

        with open('counter.txt', 'w') as f:
            f.write('%d' % indx)

print("\n✅ Final Summary:")
print(f"  ✅ Matched:   {matched}")
print(f"  ⚠️ Unmatched: {unmatched}")
print(f"  ❌ Errors:    {len(errors)}")

# ✅ Save final outputs after full completion
pd.DataFrame(fact_set_records).to_csv('matched_records.csv', index=False)
print("📁 Final full results written to matched_records.csv")

# 🧼 Optional cleanup
if os.path.exists('restart_index.txt'):
    os.remove('restart_index.txt')

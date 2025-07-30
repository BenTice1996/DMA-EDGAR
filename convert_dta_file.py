import pandas as pd
data = pd.read_stata('PostCleaning.dta')
data.to_csv('coded_contracts_post.csv')
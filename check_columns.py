import pandas as pd

df = pd.read_parquet('all_stocks_5yr_snappy.parquet')
print(df.columns)

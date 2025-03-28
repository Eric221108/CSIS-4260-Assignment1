import pandas as pd

# Load the 100x CSV dataset
df_100x = pd.read_csv('all_stocks_5yr_100x.csv')

# Define compression methods
compression_methods = [None, 'snappy', 'gzip', 'zstd']

# Convert CSV to Parquet with each compression
for method in compression_methods:
    compression_label = method if method else 'none'
    filename = f'all_stocks_5yr_100x_{compression_label}.parquet'
    
    df_100x.to_parquet(filename, compression=method)
    print(f'Created: {filename}')

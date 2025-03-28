import pandas as pd

# Load the 10x CSV dataset
df_10x = pd.read_csv('all_stocks_5yr_10x.csv')

# Compression methods
compression_methods = [None, 'snappy', 'gzip', 'zstd']

# Convert to Parquet with all compression types
for method in compression_methods:
    compression_label = method if method else 'none'
    filename = f'all_stocks_5yr_10x_{compression_label}.parquet'
    
    df_10x.to_parquet(filename, compression=method)
    print(f'Created: {filename}')

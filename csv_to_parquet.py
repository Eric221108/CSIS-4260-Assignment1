import pandas as pd

# Load the CSV dataset into a DataFrame
df = pd.read_csv('all_stocks_5yr.csv')

# Define compression methods
compression_methods = [None, 'snappy', 'gzip', 'zstd']

# Loop through each compression method and save Parquet file
for method in compression_methods:
    compression_label = method if method else 'none'
    filename = f'all_stocks_5yr_{compression_label}.parquet'
    
    # Save DataFrame to Parquet
    df.to_parquet(filename, compression=method)
    print(f'Created: {filename}')
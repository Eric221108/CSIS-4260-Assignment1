import pandas as pd
import time
import os

# Define files
files = {
    'CSV': 'all_stocks_5yr.csv',
    'Parquet_None': 'all_stocks_5yr_none.parquet',
    'Parquet_Snappy': 'all_stocks_5yr_snappy.parquet',
    'Parquet_Gzip': 'all_stocks_5yr_gzip.parquet',
    'Parquet_Zstd': 'all_stocks_5yr_zstd.parquet'
}

# Store results
results = []

for file_type, filename in files.items():
    # Measure Write Time
    start_write = time.time()
    if file_type == 'CSV':
        df = pd.read_csv(filename)
    else:
        df = pd.read_parquet(filename)
    write_time = time.time() - start_write

    # Measure Read Time
    start_read = time.time()
    df.to_csv("temp.csv") if file_type == 'CSV' else df.to_parquet("temp.parquet")
    read_time = time.time() - start_read

    # File Size
    file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB

    # Append results
    results.append([file_type, write_time, read_time, file_size])

# Create DataFrame and save results
benchmark_df = pd.DataFrame(results, columns=["Format", "Write Time (s)", "Read Time (s)", "File Size (MB)"])
benchmark_df.to_csv("benchmark_results.csv", index=False)

# Print output
print(benchmark_df)

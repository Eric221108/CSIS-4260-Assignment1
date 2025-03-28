import pandas as pd
import time
import os

# File paths for benchmarking 100x
files = {
    'CSV': 'all_stocks_5yr_100x.csv',
    'Parquet_None': 'all_stocks_5yr_100x_none.parquet',
    'Parquet_Snappy': 'all_stocks_5yr_100x_snappy.parquet',
    'Parquet_Gzip': 'all_stocks_5yr_100x_gzip.parquet',
    'Parquet_Zstd': 'all_stocks_5yr_100x_zstd.parquet'
}

# Load the dataset for writing benchmarks
df = pd.read_csv(files['CSV'])  # Load CSV as source

# Benchmark write time, read time, and file size
results = []

for file_type, filename in files.items():
    # Measure write time
    start_time = time.time()
    if "CSV" in file_type:
        df.to_csv(filename, index=False)
    else:
        df.to_parquet(filename, compression=file_type.split('_')[-1].lower() if "Parquet" in file_type else None)
    write_time = time.time() - start_time

    # Measure read time
    start_time = time.time()
    if "CSV" in file_type:
        df_read = pd.read_csv(filename)
    else:
        df_read = pd.read_parquet(filename)
    read_time = time.time() - start_time

    # Get file size in MB
    file_size = os.path.getsize(filename) / (1024 * 1024)

    # Store results
    results.append([file_type, write_time, read_time, file_size])

# Convert results to DataFrame and print
df_results = pd.DataFrame(results, columns=["Format", "Write Time (s)", "Read Time (s)", "File Size (MB)"])
print(df_results)

# Save results to a CSV for documentation
df_results.to_csv("benchmark_100x_results.csv", index=False)
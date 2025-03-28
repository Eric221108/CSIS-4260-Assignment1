import pandas as pd

# Load your dataset (Parquet Snappy)
df = pd.read_parquet('all_stocks_5yr_snappy.parquet')

# Check clearly for null values in each column
null_summary = df.isnull().sum()

# Print the result explicitly
print("Null Values in Original Dataset:")
print(null_summary)

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_parquet('all_stocks_5yr_snappy.parquet')

# Check for NaNs before filling
print("🔍 Null Values BEFORE Filling:")
print(df.isna().sum())

# Check for infinite values before filling
print("\n🔍 Checking for Infinite Values BEFORE Filling:")
print((df == np.inf).sum())
print((df == -np.inf).sum())

# Stop execution here to analyze results
exit()

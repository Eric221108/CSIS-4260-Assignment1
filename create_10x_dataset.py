import pandas as pd

# Load original CSV file
df = pd.read_csv('all_stocks_5yr.csv')

# Concatenate dataset 10 times
df_10x = pd.concat([df] * 10, ignore_index=True)

# Save new 10x larger CSV file
df_10x.to_csv('all_stocks_5yr_10x.csv', index=False)

print('10x dataset created successfully!')

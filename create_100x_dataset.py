import pandas as pd

# Load original CSV
df = pd.read_csv('all_stocks_5yr.csv')

# Concatenate dataset 100 times to create the 100x dataset
df_100x = pd.concat([df] * 100, ignore_index=True)

# Save the 100x CSV file
df_100x.to_csv('all_stocks_5yr_100x.csv', index=False)

print('100x dataset created successfully!')

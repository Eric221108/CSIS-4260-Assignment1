import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import time
import joblib  # Import joblib for saving the model and dataset

# Load dataset
df = pd.read_parquet('all_stocks_5yr_snappy.parquet')

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Add technical indicators
df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
df['RSI'] = df['close'].diff().rolling(14).apply(
    lambda x: 100 - (100 / (1 + (x[x > 0].sum() / abs(x[x < 0].sum())))), raw=False)
df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['BB_upper'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
df['BB_lower'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()

# Shift closing prices to predict next-day price
df['next_day_close'] = df.groupby('name')['close'].shift(-1)

# Fix missing values in indicators
num_cols = ['open', 'high', 'low', 'SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
df[num_cols] = df[num_cols].fillna(method='ffill').fillna(method='bfill')
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Remove rows where `next_day_close` is NaN
df = df.dropna(subset=['next_day_close'])

# Final Check: Print if any NaNs remain
print("\nFinal Null Check After Fix:\n", df.isna().sum())

# Define features and target
features = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
X = df[features]
y = df['next_day_close']

# Debug: Check for NaNs in Features and Target
print("\nNull Values in Features (X) Before Training:\n", X.isna().sum())
print("\nNull Values in Target (y) Before Training:\n", y.isna().sum())

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
start_time = time.time()
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
linear_time = time.time() - start_time
print(f"\nLinear Regression completed in {linear_time:.2f}s")

# Train Optimized Random Forest Model
start_time = time.time()
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_time = time.time() - start_time
print(f"\nRandom Forest Processing Time: {rf_time:.2f}s")

# Save the trained Random Forest model and dataset
joblib.dump({"model": rf_model, "X_train": X_train, "name": df["name"]}, "random_forest_model.pkl")
print(" Random Forest model & dataset saved as 'random_forest_model.pkl'")

# Performance Results
print("\n--- Linear Regression ---")
print("MSE:", mean_squared_error(y_test, linear_predictions))
print("MAE:", mean_absolute_error(y_test, linear_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, linear_predictions)))

print("\n--- Random Forest Regressor ---")
print("MSE:", mean_squared_error(y_test, rf_predictions))
print("MAE:", mean_absolute_error(y_test, rf_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_predictions)))

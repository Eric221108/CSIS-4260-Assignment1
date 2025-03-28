import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import time

# Load dataset
df = pl.read_parquet('all_stocks_5yr_snappy.parquet')

# Convert date column
df = df.with_columns(pl.col("date").str.to_date())

# Fix Rolling Window Calculations
df = df.with_columns([
    pl.col("close").rolling_mean(window_size=20).alias("SMA_20"),
    (pl.col("close").ewm_mean(span=12) - pl.col("close").ewm_mean(span=26)).alias("MACD"),
    (pl.col("close").rolling_mean(20) + 2 * pl.col("close").rolling_std(20)).alias("BB_upper"),
    (pl.col("close").rolling_mean(20) - 2 * pl.col("close").rolling_std(20)).alias("BB_lower")
])

# Fix RSI Calculation
def compute_rsi(series):
    gains = np.where(series > 0, series, 0).sum()
    losses = np.where(series < 0, abs(series), 0).sum()

    if losses == 0:
        return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))

df = df.with_columns(
    pl.col("close").diff().rolling_sum(14).map_elements(compute_rsi, return_dtype=pl.Float64).alias("RSI")
)

# Shift closing prices for next-day prediction
df = df.with_columns(pl.col("close").shift(-1).over("name").alias("next_day_close"))

# Fix NaN values using forward & backward fill
df = df.with_columns([
    pl.col("open").fill_null(strategy="forward").fill_null(strategy="backward"),
    pl.col("high").fill_null(strategy="forward").fill_null(strategy="backward"),
    pl.col("low").fill_null(strategy="forward").fill_null(strategy="backward"),
    pl.col("SMA_20").fill_null(strategy="forward").fill_null(strategy="backward"),
    pl.col("RSI").fill_null(strategy="forward").fill_null(strategy="backward"),
    pl.col("MACD").fill_null(strategy="forward").fill_null(strategy="backward"),
    pl.col("BB_upper").fill_null(strategy="forward").fill_null(strategy="backward"),
    pl.col("BB_lower").fill_null(strategy="forward").fill_null(strategy="backward"),
])

# Remove rows where `next_day_close` is NaN
df = df.drop_nulls(subset=["next_day_close"])

# Convert Polars to Pandas for scikit-learn
df = df.to_pandas()

# Final Check for NaNs
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

# Linear Regression Model
start_time = time.time()
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
linear_time = time.time() - start_time
print(f"\nLinear Regression completed in {linear_time:.2f}s")

# Optimized Random Forest Model
start_time = time.time()
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_time = time.time() - start_time
print(f"\nRandom Forest Processing Time: {rf_time:.2f}s")

# Performance Results
print("\n--- Linear Regression ---")
print("MSE:", mean_squared_error(y_test, linear_predictions))
print("MAE:", mean_absolute_error(y_test, linear_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, linear_predictions)))

print("\n--- Random Forest Regressor ---")
print("MSE:", mean_squared_error(y_test, rf_predictions))
print("MAE:", mean_absolute_error(y_test, rf_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_predictions)))

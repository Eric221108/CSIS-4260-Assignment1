import polars as pl

# --- Load Parquet Snappy data ---
df_polars = pl.read_parquet('all_stocks_5yr_snappy.parquet')

# --- Convert 'date' to datetime ---
df_polars = df_polars.with_columns(pl.col("date").str.to_date())

# --- Sort by stock and date ---
df_polars = df_polars.sort(["name", "date"])

# --- Add Technical Indicators (Polars version) ---

# 1. Calculate SMA (20) - Simple Moving Average for 20 days
df_polars = df_polars.with_columns([
    pl.col("close").rolling_mean(window_size=20).alias("SMA_20")
])

# 2. Calculate RSI (14) - Relative Strength Index for 14 periods
df_polars = df_polars.with_columns([
    (100 - (100 / (1 + (pl.col("close").diff().clip(0, None).rolling_mean(14) /
                      (-pl.col("close").diff().clip(None, 0).rolling_mean(14)))))).alias("RSI_14")
])

# 3. Calculate MACD - Exponential Moving Averages (12-day and 26-day)
df_polars = df_polars.with_columns([
    (pl.col("close").ewm_mean(span=12) - pl.col("close").ewm_mean(span=26)).alias("MACD")
])

# 4. Calculate Bollinger Bands (20-day SMA ± 2 * Standard Deviation)
df_polars = df_polars.with_columns([
    (pl.col("close").rolling_mean(window_size=20) + 2 * pl.col("close").rolling_std(window_size=20)).alias("BB_upper"),
    (pl.col("close").rolling_mean(window_size=20) - 2 * pl.col("close").rolling_std(window_size=20)).alias("BB_lower")
])

# --- Save the updated Polars dataset ---
df_polars.write_parquet('technical_all_stocks_5yr_polars.parquet')

print(" Features added! New dataset saved as 'technical_all_stocks_5yr_polars.parquet'")

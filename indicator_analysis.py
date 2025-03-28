import pandas as pd
import polars as pl
import ta
import time

# --- Load Parquet Snappy data (optimized file chosen from Part 1) ---
df_pandas = pd.read_parquet('all_stocks_5yr_snappy.parquet').dropna(subset=['open', 'high', 'low'])
df_polars = pl.read_parquet('all_stocks_5yr_snappy.parquet').drop_nulls(subset=['open', 'high', 'low'])

# --- Convert date columns ---
df_pandas['date'] = pd.to_datetime(df_pandas['date'])
df_polars = df_polars.with_columns(pl.col('date').str.to_date())

# --- Pandas indicators function ---
def indicators_pandas(df):
    df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['RSI'] = ta.momentum.rsi(df['close'], window=14, fillna=True)
    macd_indicator = ta.trend.MACD(df['close'], fillna=True)
    df['MACD'] = macd_indicator.macd()
    bb_indicator = ta.volatility.BollingerBands(df['close'], fillna=True)
    df['BB_upper'] = bb_indicator.bollinger_hband()
    df['BB_lower'] = bb_indicator.bollinger_lband()
    return df

# --- Pandas benchmarking ---
start_time = time.time()
df_pandas_final = df_pandas.groupby('name', group_keys=False).apply(indicators_pandas, include_groups=False)
pandas_time = time.time() - start_time
print(f"Pandas Processing Time: {pandas_time:.2f}s")

# --- Clearly print columns and null summary (Pandas) ---
print("\nPandas Columns:", df_pandas_final.columns.tolist())
print("Pandas Null Values:\n", df_pandas_final.isnull().sum())

# --- Polars indicators function (explicit column retention) ---
def indicators_polars(df):
    df_pd = df.to_pandas()

    # Calculate indicators explicitly
    df_pd['SMA_20'] = df_pd['close'].rolling(window=20, min_periods=1).mean()
    df_pd['RSI'] = ta.momentum.rsi(df_pd['close'], window=14, fillna=True)
    macd_indicator = ta.trend.MACD(df_pd['close'], fillna=True)
    df_pd['MACD'] = macd_indicator.macd()
    bb_indicator = ta.volatility.BollingerBands(df_pd['close'], fillna=True)
    df_pd['BB_upper'] = bb_indicator.bollinger_hband()
    df_pd['BB_lower'] = bb_indicator.bollinger_lband()

    # Explicitly select all columns clearly
    columns_order = ['date', 'open', 'high', 'low', 'close', 'volume', 'name',
                     'SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
    
    return pl.from_pandas(df_pd[columns_order])

# --- Polars benchmarking clearly ---
df_polars = df_polars.sort(['name', 'date'])

start_time = time.time()
df_polars_final = df_polars.group_by('name').map_groups(indicators_polars)
polars_time = time.time() - start_time
print(f"\nPolars Processing Time: {polars_time:.2f}s")

# --- Explicitly display columns and null counts (Polars) ---
pl.Config.set_tbl_cols(len(df_polars_final.columns))
print("\nPolars Columns:", df_polars_final.columns)
print("Polars Null Count:\n", df_polars_final.null_count())

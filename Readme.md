# CSIS 4260 – Assignment 1 (Part 1: Storing and Retrieving Data)

## Overview
In this assignment, I benchmarked two data formats (CSV vs. Parquet) with various compression methods. I tested reading speeds and file sizes for a stock market dataset at scales 1x (29MB), 10x (~290MB), and 100x (~2.9GB).

---

## Virtual Environment
I used Python's built-in `venv` tool for simplicity and reliability.

### Steps to Activate the Environment:
```bash
source myenv/bin/activate
```

### Install Dependencies
Use the following command to install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Benchmark Results

### 📌 1x Dataset (~29MB)
| Format              | Load Time (seconds) | File Size (MB) |
|---------------------|---------------------|----------------|
| CSV                 | 0.16                | 28.70          |
| Parquet (None)      | 0.06                | 12.73          |
| Parquet (Snappy)    | 0.03                | 10.15          |
| Parquet (Gzip)      | 0.03                | 8.06           |
| Parquet (Zstd)      | 0.03                | 8.09           |

### Recommendation (1x scale)
- **Parquet (Snappy or Gzip)**: Both fast, but Gzip has smaller file size, ideal for storage.

---

### 10x Dataset (~290MB)
| Format              | Load Time (seconds) | File Size (MB) |
|---------------------|---------------------|----------------|
| CSV                 | 1.51                | 282.10         |
| Parquet (None)      | 0.25                | 118.02         |
| Parquet (Snappy)    | 0.22                | 95.35          |
| Parquet (Gzip)      | 0.27                | 75.97          |
| Parquet (Zstd)      | 0.22                | 75.43          |

### Recommendation for 10x
- **Parquet (Snappy)**: Fastest load, excellent compression balance, perfect for quick analytics.

---

### 100x Dataset (~2.9GB)
| Format              | Load Time (seconds) | File Size (MB) |
|---------------------|---------------------|----------------|
| CSV                 | 17.15               | 2821.02        |
| Parquet (None)      | 3.37                | 1178.39        |
| Parquet (Snappy)    | 3.06                | 951.67         |
| Parquet (Gzip)      | 3.55                | 758.09         |
| Parquet (Zstd)      | 3.00                | 751.57         |

### Recommendation for 100x
- **Parquet (Zstd)**: Fastest loading time with significant compression; best for large-scale data processing.

---

##  Final Conclusion
- CSV becomes slow and storage-intensive at scale.
- **Parquet with Snappy** is best for small-to-medium scales (1x, 10x).
- **Parquet with Zstd** clearly wins at the largest scale (100x), offering the optimal blend of speed and storage.

Overall, **Parquet** format is highly recommended for efficient data storage and retrieval, particularly for growing datasets.


---

## **(Part 2: Data Manipulation, Analysis & Model Building)**

### **Overview**  
In this part of the assignment, I compared two dataframe libraries, **Pandas vs. Polars**, for:  
- **Enhancing the dataset with technical indicators**  
- **Predicting next-day closing prices** using machine learning models  
- **Comparing speed and performance** of both libraries  

## **Enhancing the Dataset with Technical Indicators**  

The dataset was enhanced by adding the following four **technical indicators** for each stock:

| **Indicator**  | **Formula Used** |
|------------|-------------|
| **SMA (20)** | 20-day Simple Moving Average |
| **RSI (14)** | Relative Strength Index based on price momentum |
| **MACD** | Exponential Moving Averages (12-day & 26-day) |
| **Bollinger Bands** | 20-day SMA ± (2 * Standard Deviation) |

These indicators were added using both **Pandas and Polars** before proceeding with predictions.

---

## **Handling Missing Values**  

| **Method** | **Pandas Approach** | **Polars Approach** |
|--------|----------------|-----------------|
| **Forward Fill** | `df.ffill()` | `.fill_null(strategy="forward")` |
| **Backward Fill** | `df.bfill()` | `.fill_null(strategy="backward")` |
| **Mean Fill** | `df.fillna(df.mean())` | `.fill_null(pl.col().mean())` |

Both libraries successfully **filled missing values** instead of dropping them.

---

### **Comparison: Dropping vs. Filling Nulls**  

| **Method** | **Pandas Result** | **Polars Result** |
|--------|--------------|--------------|
| **Dropping Nulls** | Faster but loses data | Faster but loses data |
| **Filling Nulls** | Slightly slower, but preserves data & accuracy | Slightly slower, but preserves data & accuracy |

### **Final Decision**  
Filling missing values led to **better prediction accuracy**, so this method was used in both Pandas and Polars implementations.

---

## **Machine Learning Models Used**  

Two models were selected to predict the **next day's closing price**:

| **Model** | **Purpose** |
|------------|-------------|
| **Linear Regression** | Predicts stock price trends with a simple approach |
| **Random Forest Regressor** | Captures **complex patterns** in stock prices |

The dataset was split **80-20** for training and testing.

---

## **Performance Comparison: Pandas vs. Polars**  

### **Execution Time**  
| **Library** | **Execution Time (Seconds)** |
|---------|--------------------------|
| **Pandas** | 56.39s |
| **Polars** | 55.00s |

### **Prediction Accuracy (MSE, MAE, RMSE)**  
| **Metric** | **Pandas** | **Polars** |
|--------|--------|--------|
| **Linear Regression MSE** | 4.2958 | 4.2959 |
| **Linear Regression MAE** | 0.8524 | 0.8525 |
| **Linear Regression RMSE** | 2.0726 | 2.0727 |
| **Random Forest MSE** | 4.5780 | 4.5518 |
| **Random Forest MAE** | 0.8681 | 0.8678 |
| **Random Forest RMSE** | 2.1396 | 2.1335 |

---

## **Key Observations**  
1. **Polars is slightly faster than Pandas**  
   - Polars reduced processing time by **1.4 seconds**.  
   - This speed difference **matters more at large dataset scales (10x, 100x).**  

2. **Prediction Accuracy is almost identical**  
   - Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are nearly **the same** in both libraries.  

3. **Polars is better suited for large-scale data**  
   - **More memory-efficient** compared to Pandas.  
   - Better handling of **multi-threading for faster computations**.  

---

## **Final Recommendation**  

### **For Small Datasets (1x size) → Use Pandas**  
- **Easier to use**, familiar syntax.  
- **Integrates well with existing ML libraries**.  

### **For Large Datasets (10x, 100x) → Use Polars**  
- **More efficient for big datasets**.  
- **Faster processing** and **lower memory usage**.  

**Final Choice:** **Polars is better for future scalability.**  


---

# **(Part 3: Creating a Visual Dashboard)**

## **Overview**  
In this part, I created **interactive dashboards** to visualize:
- **Benchmarking results** (Part A) for CSV vs. Parquet formats across different dataset sizes.
- **Stock price predictions** (Part B) using machine learning models.  

We tested **two dashboarding libraries:**
1. **Streamlit** (easy setup, but less customizable)
2. **Dash** (more flexible, better UI, but requires more code)  

After comparison, **Dash was chosen** for the final price prediction dashboard.

---
# ** Part A: Benchmarking Dashboard (Streamlit & Dash)**  

## **Goal**  
To visualize the **benchmarking results** from **Part 1** using both **Streamlit** and **Dash**, allowing users to:
- Compare **read/write speeds** for CSV vs. Parquet (None, Snappy, Gzip, Zstd).
- Analyze file size differences across different dataset scales (1x, 10x, 100x).

## **Implementation**  
- **Dataset:** `benchmark_results.csv`  
- **Libraries Used:** `dash`, `streamlit`, `plotly`, `pandas`  
- **User Interaction:** Dataset scale selection (1x, 10x, 100x)  

## **Comparison of Streamlit vs. Dash**  

| **Feature**      | **Streamlit**  | **Dash** |
|-----------------|---------------|----------|
| **Ease of Setup** |  Very simple |  More setup required |
| **Customization** |  Limited |  More control over UI |
| **Performance** |  Lightweight | More scalable |
| **Interactivity** |  Limited |  More flexible |

### **Final Decision**  
Since **Dash** offers more flexibility and better UI customization, we chose it for **Part B** (Price Prediction Dashboard).

---

# ** Part B: Stock Price Prediction Dashboard (Dash)**  

## **Goal**  
To build an **interactive stock price prediction dashboard** that:
- Displays **historical stock prices** 
- Shows **predicted future prices** using a **machine learning model** 
- Allows users to **select a stock ticker** and **update the charts dynamically**.

## **Implementation**  
- **Dataset Used:** `all_stocks_5yr_snappy.parquet`
- **Prediction Model Used:** `Random Forest Regressor (trained in Part 2)`
- **Libraries Used:** `dash`, `plotly`, `pandas`, `scikit-learn`

## **Dashboard Features**  
| **Feature** | **Implementation** |
|------------|--------------------|
| **Stock Selection** | Dropdown menu for selecting company ticker |
| **Historical Prices** | Interactive **line chart** using Plotly |
| **Price Prediction** | Uses **Random Forest Model** (from Part 2) |

### **How Predictions Work**
1. **User selects a stock ticker** from the dropdown.
2. **Dashboard loads historical prices** for the selected stock.
3. **Random Forest Model predicts next 30 days' closing prices**.
4. **Charts update dynamically**.

---

## ** Final Thoughts**  

### **Key Learnings**
- **Dash is more powerful** than Streamlit for creating detailed, interactive dashboards.
- **Parquet (Snappy) is the best format** for working with large datasets.
- **Random Forest Regressor outperforms Linear Regression** in predicting stock prices.
- **Polars is faster than Pandas** for large-scale data analysis.


### **Final Submission**
This assignment successfully benchmarks data formats, analyzes stock data, and builds interactive dashboards.  
 **Final Choice:** **Dash + Parquet (Snappy) + Random Forest Regressor** 🚀

---


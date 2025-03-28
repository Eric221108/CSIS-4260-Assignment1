import streamlit as st
import pandas as pd

# Load benchmark results
benchmark_1x = pd.read_csv("benchmark_results.csv")
benchmark_10x = pd.read_csv("benchmark_10x_results.csv")
benchmark_100x = pd.read_csv("benchmark_100x_results.csv")

st.title("📊 Benchmark Results for CSV vs. Parquet")

# Select dataset size
dataset_choice = st.radio("Select Dataset Scale:", ["1x (~29MB)", "10x (~290MB)", "100x (~2.9GB)"])

# Load the selected dataset
if dataset_choice == "1x (~29MB)":
    data = benchmark_1x
elif dataset_choice == "10x (~290MB)":
    data = benchmark_10x
else:
    data = benchmark_100x

# Display data table
st.write("### 📌 Benchmark Data")
st.dataframe(data)

# Create bar charts
st.write("### 📊 Read & Write Time Comparison")
st.bar_chart(data.set_index("Format")[["Write Time (s)", "Read Time (s)"]])

st.write("### 🗄️ File Size Comparison")
st.bar_chart(data.set_index("Format")["File Size (MB)"])

import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Load benchmark results
benchmark_1x = pd.read_csv("benchmark_results.csv")
benchmark_10x = pd.read_csv("benchmark_10x_results.csv")
benchmark_100x = pd.read_csv("benchmark_100x_results.csv")

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("📊 Benchmarking Dashboard: Streamlit vs Dash"),
    
    dcc.Dropdown(
        id="dataset_choice",
        options=[
            {"label": "1x Dataset", "value": "1x"},
            {"label": "10x Dataset", "value": "10x"},
            {"label": "100x Dataset", "value": "100x"},
        ],
        value="1x"
    ),

    dcc.Graph(id="benchmark_chart"),
    html.Div(id="file_size_table")
])

@app.callback(
    dash.dependencies.Output("benchmark_chart", "figure"),
    dash.dependencies.Output("file_size_table", "children"),
    [dash.dependencies.Input("dataset_choice", "value")]
)
def update_dashboard(dataset_choice):
    if dataset_choice == "1x":
        data = benchmark_1x
    elif dataset_choice == "10x":
        data = benchmark_10x
    else:
        data = benchmark_100x

    # Bar Chart
    fig = px.bar(data, x="Format", y=["Read Time (s)", "Write Time (s)"],
                 title=f"Read & Write Performance - {dataset_choice} Scale",
                 barmode="group")

    # File Size Table
    table = html.Table([
        html.Thead(html.Tr([html.Th("Format"), html.Th("File Size (MB)")]))] +
        [html.Tr([html.Td(row["Format"]), html.Td(row["File Size (MB)"])]) for _, row in data.iterrows()]
    )

    return fig, table

# Run Command: python dashboard_benchmark_dash.py
if __name__ == "__main__":
    app.run_server(debug=True)

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the dataset (Using Snappy Parquet format for efficiency)
df = pd.read_parquet("all_stocks_5yr_snappy.parquet")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Extract unique company tickers
company_tickers = df["name"].unique()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define Layout
app.layout = dbc.Container([
    html.H1("Stock Price Prediction Dashboard", className="text-center mt-3"),

    # Dropdown for selecting company ticker
    dbc.Row([
        dbc.Col(html.Label("Select a Company:"), width=2),
        dbc.Col(dcc.Dropdown(
            id="company-dropdown",
            options=[{"label": ticker, "value": ticker} for ticker in company_tickers],
            value=company_tickers[0],  # Default selection
            clearable=False
        ), width=6)
    ], className="mb-4"),

    # Line chart for historical prices
    dcc.Graph(id="price-chart"),

    # Placeholder for predictions (to be updated)
    dcc.Graph(id="prediction-chart")
])

# Define Callbacks
@app.callback(
    [Output("price-chart", "figure"), Output("prediction-chart", "figure")],
    [Input("company-dropdown", "value")]
)
def update_graphs(selected_company):
    # Filter data for selected company
    company_data = df[df["name"] == selected_company]

    # Historical price chart
    fig_price = px.line(company_data, x="date", y="close", title=f"{selected_company} - Historical Prices")

    # TODO: Replace with actual model predictions
    # Placeholder: last 30 days repeated as future predictions
    future_dates = pd.date_range(start=company_data["date"].max(), periods=30, freq="D")
    future_prices = [company_data["close"].iloc[-1]] * 30  

    fig_pred = px.line(x=future_dates, y=future_prices, title=f"{selected_company} - Predicted Prices")

    return fig_price, fig_pred

# Run the Dash App
if __name__ == "__main__":
    app.run_server(debug=True)

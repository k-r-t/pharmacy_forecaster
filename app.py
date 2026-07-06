import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Pharmacy Demand Forecaster",
    layout="wide"
)

st.markdown("""
<style>
.stApp{
    background-color:#E0F8E0;
}
.header-box{
    background-color:#70A970;
    padding:20px;
    border-radius:10px;
    text-align:center;
    color:white;
}
.metric-box{
    background-color:white;
    padding:10px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="header-box">
        <h1>Pharmacy Demand Forecaster</h1>
    </div>
    """,
    unsafe_allow_html=True
)

file_map = {
    "Daily": "data/archive (2)/salesdaily.csv",
    "Weekly": "data/archive (2)/salesweekly.csv",
    "Monthly": "data/archive (2)/salesmonthly.csv",
    "Hourly": "data/archive (2)/saleshourly.csv"
}

ATC_COLUMNS = [
    "M01AB", "M01AE", "N02BA", "N02BE",
    "N05B", "N05C", "R03", "R06"
]


@st.cache_data
def load_sample_data(path):
    df = pd.read_csv(path)
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
        df = df.dropna(subset=["datum"])
    return df


@st.cache_data
def load_uploaded_data(file):
    df = pd.read_csv(file)
    return df

st.sidebar.header("Navigation")

data_source = st.sidebar.radio(
    "Select Data Source",
    ["Use Sample Dataset", "Upload Your Own CSV"]
)

model_choice = st.sidebar.radio(
    "Select Analysis Model",
    ["Prophet", "ARIMA"]
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon", 7, 60, 30
)

df = None
date_col = None
value_col = None
dataset_label = ""

if data_source == "Use Sample Dataset":
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset",
        ["Daily", "Weekly", "Monthly", "Hourly"]
    )
    selected_drug = st.sidebar.selectbox(
        "Select Drug Category",
        ATC_COLUMNS
    )

    df = load_sample_data(file_map[dataset_choice])
    date_col = "datum"
    value_col = selected_drug
    dataset_label = f"{dataset_choice} Sample Data"

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = load_uploaded_data(uploaded_file)

        st.sidebar.markdown("**Map Your Columns**")

        date_col = st.sidebar.selectbox(
            "Select Date Column",
            df.columns
        )
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.sidebar.error("No numeric columns found in this file.")
        value_col = st.sidebar.selectbox(
            "Select Value Column to Forecast",
            numeric_cols if numeric_cols else df.columns
        )

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        dataset_label = uploaded_file.name
    else:
        st.info("Upload a CSV file from the sidebar to get you own forecast.")
        st.stop()

# VALIDATION
if df is None or df.empty:
    st.error("No valid data available. Please check your file or selections.")
    st.stop()

if value_col is None or date_col is None:
    st.error("Please select a valid date and value column.")
    st.stop()

st.subheader(f"{model_choice} Analysis : {value_col} ({dataset_label})")

with st.expander("Preview Data"):
    st.dataframe(df.head(20))

# PROPHET
if model_choice == "Prophet":
    prophet_df = df[[date_col, value_col]].rename(
        columns={date_col: "ds", value_col: "y"}
    )
    prophet_df = prophet_df.dropna()

    if len(prophet_df) < 2:
        st.error("Not enough data points to run a forecast. Please choose a different column or upload more data.")
        st.stop()

    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    display_df = forecast[["ds", "yhat"]].rename(
        columns={"ds": "Date Stamp", "yhat": "Predicted Demand"}
    )

    fig = px.line(
        display_df,
        x="Date Stamp",
        y="Predicted Demand",
        color_discrete_sequence=["red"]
    )
    fig.add_scatter(
        x=prophet_df["ds"],
        y=prophet_df["y"],
        mode="lines",
        name="Actual Sales",
        line=dict(color="blue")
    )
    fig.update_layout(
        xaxis_title="Date Stamp",
        yaxis_title="Predicted Demand"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Values")
    st.dataframe(display_df.tail(forecast_days))

    csv = display_df.tail(forecast_days).to_csv(index=False)
    st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")

# ARIMA 
elif model_choice == "ARIMA":
    arima_series = (
        df.set_index(date_col)[value_col]
        .dropna()
        .sort_index()
    )

    if len(arima_series) < 5:
        st.error("Not enough data points to run ARIMA. Please choose a different column or upload more data.")
        st.stop()

    model = ARIMA(arima_series, order=(5, 1, 0))
    results = model.fit()

    forecast_values = results.forecast(steps=forecast_days)

    forecast_df = pd.DataFrame({
        "Date Stamp": forecast_values.index,
        "Predicted Demand": forecast_values.values
    })

    fig = px.line(
        forecast_df,
        x="Date Stamp",
        y="Predicted Demand",
        color_discrete_sequence=["red"]
    )
    fig.add_scatter(
        x=arima_series.index,
        y=arima_series.values,
        mode="lines",
        name="Actual Sales",
        line=dict(color="blue")
    )
    fig.update_layout(
        xaxis_title="Date Stamp",
        yaxis_title="Predicted Demand"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Values")
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv(index=False)
    st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")

st.markdown("---")
st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset", dataset_label)

with col2:
    st.metric("Value Column", value_col)

with col3:
    st.metric("Forecast Horizon", f"{forecast_days} periods")

import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Pharmacy Inventory Forecaster", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #E0F8E0; }
    h1 { background-color: #70A970; color: white; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🏥 Pharmacy Inventory Forecaster</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    file_path = 'data/archive (2)/salesdaily.csv'
    df = pd.read_csv(file_path)
    
    if 'Weekday Name' in df.columns:
        df = df.drop(columns=['Weekday Name'])

    df.columns.values[0] = 'datum'
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    df = df.dropna(subset=['datum'])
    
    return df

try:
    df = load_data()
    columns = [col for col in df.columns if col != 'datum']
    
    st.sidebar.header("Configuration")
    selected_drug = st.sidebar.selectbox("Select Medication Category:", columns)
    days = st.sidebar.slider("Days to Forecast:", 7, 90, 30)

    df_prophet = df[['datum', selected_drug]].rename(columns={'datum': 'ds', selected_drug: 'y'})
    
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    st.subheader(f"Analyzing Sales for: {selected_drug}")
    fig = px.line(forecast, x='ds', y='yhat', title="AI Demand Forecast")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"DATA ERROR: {e}")

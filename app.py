import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

st.set_page_config(page_title="Pharmacy Inventory System", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #E0F8E0; }
    .header-box { background-color: #70A970; color: white; padding: 20px; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-box"><h1>🏥 Pharmacy Demand Forecaster</h1></div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    file_path = 'data/archive (2)/salesdaily.csv'
    df = pd.read_csv(file_path)
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    if 'datum' in df.columns:
        df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    
     for col in df.columns:
        if col != 'datum':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
      df = df.dropna(subset=['datum'])
    return df

try:
    df = load_data()
    drug_cols = [c for c in df.columns if c != 'datum' and pd.api.types.is_numeric_dtype(df[c])]

    st.sidebar.header("Navigation")
    model_choice = st.sidebar.radio("Select Analysis Model:", ["Prophet", "ARIMA"])
    
    st.sidebar.header("Configuration")
    selected_drug = st.sidebar.selectbox("Select Medication/Frequency:", drug_cols)
    days = st.sidebar.slider("Days to Forecast:", 7, 60, 30)

     if model_choice == "Prophet":
        st.subheader(f"Analyzing Sales with Prophet: {selected_drug}")
        df_prophet = df[['datum', selected_drug]].rename(columns={'datum': 'ds', selected_drug: 'y'})
        
        model = Prophet(daily_seasonality=True).fit(df_prophet)
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        
        fig = px.line(forecast, x='ds', y='yhat', color_discrete_sequence=['red'])
        fig.add_scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual Sales', line=dict(color='blue'))
        fig.update_xaxes(tickformat="%a, %b %d")
        st.plotly_chart(fig, use_container_width=True)

    elif model_choice == "ARIMA":
        st.subheader(f"Analyzing Sales with ARIMA: {selected_drug}")
      
        df_arima = df.set_index('datum')[selected_drug].sort_index()
        model = ARIMA(df_arima, order=(5,1,0))
        results = model.fit()
        forecast_values = results.forecast(steps=days)
        
        forecast_df = pd.DataFrame({'ds': forecast_values.index, 'yhat': forecast_values.values})
       
        fig = px.line(forecast_df, x='ds', y='yhat', color_discrete_sequence=['red'])
        fig.add_scatter(x=df_arima.index, y=df_arima.values, mode='lines', name='Actual Sales', line=dict(color='blue'))
        fig.update_xaxes(tickformat="%a, %b %d")
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"SYSTEM ERROR: {e}")

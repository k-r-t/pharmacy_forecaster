import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px


st.set_page_config(page_title="Pharmacy Inventory Forecaster", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #E0F8E0; }
    [data-testid="stSidebar"] { background-color: #E0F8E0; }
    .header-box { 
        background-color: #70A970; 
        color: white; 
        padding: 20px; 
        border-radius: 10px; 
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-box"><h1>🏥 Pharmacy Inventory Forecaster</h1></div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    file_path = 'data/archive (2)/salesdaily.csv'

    df = pd.read_csv(file_path, index_col=0)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    if 'datum' not in df.columns:
        df = df.reset_index()
        
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    df = df.dropna(subset=['datum'])
    return df
try:
    df = load_data()
    drug_cols = [c for c in df.columns if c != 'datum']
    
    st.sidebar.header("Configuration")
    selected_drug = st.sidebar.selectbox("Select Medication Category:", drug_cols)
    days = st.sidebar.slider("Days to Forecast:", 7, 90, 30)

    df_prophet = df[['datum', selected_drug]].rename(columns={'datum': 'ds', selected_drug: 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    st.subheader(f"Analyzing Sales for: {selected_drug}")
    
    forecast_df = forecast.rename(columns={'yhat': 'Predicted Sales'})
    
    fig = px.line(forecast_df, x='ds', y='Predicted Sales', color_discrete_sequence=['red'])
    fig.add_scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual Sales', line=dict(color='blue'))
    
    fig.update_layout(title="AI Demand Forecast", xaxis_title="Date", yaxis_title="Sales Quantity")
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"SYSTEM ERROR: {e}")

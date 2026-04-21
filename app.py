import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Pharmacy Inventory Forecaster", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #E0F8E0; }
    
    /* Sidebar Background (Configuration) */
    [data-testid="stSidebar"] {
        background-color: #E0F8E0;
    }
    
    .sidebar-title { 
        background-color: #70A970; 
        color: white; 
        padding: 15px; 
        border-radius: 10px; 
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    file_path = 'data/archive (2)/salesdaily.csv'

    df = pd.read_csv(file_path)
    if 'Weekday Name' in df.columns:
        df = df.drop(columns=['Weekday Name'])
       df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.dropna(subset=[df.columns[0]])
    df = df.rename(columns={df.columns[0]: 'datum'})
    
    return df

st.sidebar.markdown("<div class='sidebar-title'> Pharmacy Inventory Forecaster</div>", unsafe_allow_html=True)

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
    fig.add_scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual Sales', line=dict(color='blue'))
    fig.update_traces(line_color='red', selector=dict(name='yhat'))
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"SYSTEM ERROR: {e}")
    st.warning("If this error persists, you MUST delete the app in your Streamlit Cloud dashboard and reconnect the GitHub repository to clear the cached memory.")

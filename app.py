import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Pharmacy Inventory Forecaster", layout="wide")

st.markdown("<h1 style='text-align: left; color: #007BFF;'> Pharmacy Inventory Forecaster</h1>", unsafe_allow_html=True)
st.markdown("---")

# 3. Robust Data Loader
@st.cache_data
def load_data():
    file_path = 'data/archive (2)/salesdaily.csv'
    # 'encoding' helps with special characters, 'skipinitialspace' cleans the CSV reading
    df = pd.read_csv(file_path, skipinitialspace=True, encoding='utf-8')
    
    # Strip whitespace from column names (e.g., " datum " becomes "datum")
    df.columns = df.columns.str.strip()
    
    # Force convert to date, turning "Thursday" into "Not a Time" (NaT)
    # This prevents the app from crashing.
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')
    
    # Remove the bad rows (where date is now NaT)
    df = df.dropna(subset=['datum'])
    
    return df

try:
    # Load and show the data to verify it works
    df = load_data()
    
    # Filter columns to exclude 'datum' for the medication selection
    columns = [col for col in df.columns if col != 'datum']
    
    # Sidebar Controls
    st.sidebar.header("Configuration")
    selected_drug = st.sidebar.selectbox("Select Medication Category:", columns)
    days = st.sidebar.slider("Days to Forecast:", 7, 90, 30)

    # Prepare data for Prophet
    df_prophet = df[['datum', selected_drug]].rename(columns={'datum': 'ds', selected_drug: 'y'})
    
    # Train Model
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    # RENAME COLUMNS
    display_forecast = forecast.rename(columns={
        'ds': 'Date', 
        'yhat': 'Predicted Sales',
        'yhat_lower': 'Lower Bound',
        'yhat_upper': 'Upper Bound'
    })

    # 4. Display Layout
    st.subheader(f"Analyzing Sales for: {selected_drug}")
    
    # Professional Plotly Chart
    fig = px.line(display_forecast, x='Date', y='Predicted Sales', title="AI Demand Forecast", color_discrete_sequence=['red'])
    fig.add_scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual Sales', line=dict(color='blue'))
    fig.update_layout(template="plotly_white", hovermode="x unified")

    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.info("💡 **Insight:** The red line represents the AI forecast, while the blue line shows historical data.")
        st.download_button(
            label="📥 Download Forecast Data", 
            data=display_forecast.to_csv(index=False).encode('utf-8'), 
            file_name="pharmacy_forecast.csv", 
            mime="text/csv"
        )

except Exception as e:
    st.error(f"SYSTEM ERROR: {e}")
    st.write("If you see this, please delete the app from your Streamlit dashboard and reconnect your GitHub repository.")

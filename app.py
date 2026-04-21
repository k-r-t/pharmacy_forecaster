import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Pharmacy Inventory Forecaster", layout="wide")

st.markdown(
    "<h1 style='text-align: left; color: #007BFF;'> Pharmacy Inventory Forecaster</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")


@st.cache_data
def load_data():
    file_path = "data/archive (2)/salesdaily.csv"
    df = pd.read_csv(file_path)
    df["datum"] = pd.to_datetime(df["datum"])
    return df


try:
    df = load_data()
    columns = [col for col in df.columns if col != "datum"]

    st.sidebar.header("Configuration")
    selected_drug = st.sidebar.selectbox("Select Medication Category:", columns)
    days = st.sidebar.slider("Days to Forecast:", 7, 90, 30)

    df_prophet = df[["datum", selected_drug]].rename(
        columns={"datum": "ds", selected_drug: "y"}
    )

    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    display_forecast = forecast.rename(
        columns={
            "ds": "Date",
            "yhat": "Predicted Sales",
            "yhat_lower": "Lower Bound",
            "yhat_upper": "Upper Bound",
        }
    )

    st.subheader(f"Analyzing Sales for: {selected_drug}")

    fig = px.line(
        display_forecast,
        x="Date",
        y="Predicted Sales",
        title="AI Demand Forecast",
        color_discrete_sequence=["red"],
    )
    fig.add_scatter(
        x=df_prophet["ds"],
        y=df_prophet["y"],
        mode="lines",
        name="Actual Sales",
        line=dict(color="blue"),
    )
    fig.update_layout(template="plotly_white", hovermode="x unified")

    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.info(
            "💡Insight: The red line represents the AI forecast, while the blue line shows historical data."
        )
        st.markdown(
            "Instructions:Adjust the days to forecast using the sidebar slider to see how the projections shift."
        )

        csv = display_forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast Data",
            data=csv,
            file_name="pharmacy_forecast.csv",
            mime="text/csv",
        )

except Exception as e:
    st.error(
        f"Error loading file: {e}. Please check that your CSV is in the correct 'data/archive (2)/' folder."
    )

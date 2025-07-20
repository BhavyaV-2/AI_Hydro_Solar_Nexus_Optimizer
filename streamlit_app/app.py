import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Hydro-Solar Nexus Arbitrageur",
    page_icon="🌊☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_all_models_and_data():
    print("Loading all AI models and data assets...")
    try:
        solar_model = tf.keras.models.load_model('assets/cnn_lstm_attention_model.keras')
        price_model = xgb.XGBRegressor()
        price_model.load_model('assets/price_forecaster.json')
        df_master = pd.read_csv('assets/Master_Indian_Data.csv', parse_dates=['datetime'], index_col='datetime')
        return solar_model, price_model, df_master
    except Exception as e:
        st.error(f"Fatal Error Loading Assets: {e}")
        return None, None, None

solar_model, price_model, df_master = load_all_models_and_data()

def get_optimal_schedule(solar_forecast, price_forecast, initial_storage, max_storage, flood_level_pct, drought_level_pct, max_release_per_hour, mwh_per_mcm):
    schedule, actions, storage_trace = [], [], [initial_storage]
    current_storage = initial_storage
    flood_level_mcm = max_storage * flood_level_pct
    drought_level_mcm = max_storage * drought_level_pct
    for i in range(24):
        price_now, solar_now = price_forecast[i], solar_forecast[i]
        if current_storage > flood_level_mcm: action, release = "GENERATE_MAX_FLOOD", max_release_per_hour
        elif price_now > np.percentile(price_forecast, 75) and (solar_forecast.max() == 0 or solar_now < (solar_forecast.max()*0.1)): action, release = "GENERATE_PEAK", max_release_per_hour
        elif solar_now > np.percentile(solar_forecast, 75) and current_storage < flood_level_mcm: action, release = "HOLD_STORE", 0
        else: action, release = "HOLD_CONSERVE", 0
        if current_storage - release < drought_level_mcm:
            release = max(0, current_storage - drought_level_mcm)
            if action not in ["GENERATE_MAX_FLOOD"]: action = "HOLD_DROUGHT_RISK"
        current_storage -= release
        actions.append(action); schedule.append(release); storage_trace.append(current_storage)
    hydropower_mwh = np.array(schedule) * mwh_per_mcm
    revenue = np.sum(hydropower_mwh * price_forecast)
    return actions, hydropower_mwh, revenue, np.array(storage_trace)

st.sidebar.title("Simulation Controls")
st.sidebar.markdown("Define the 24-hour scenario to be optimized:")
weather_scenario = st.sidebar.selectbox("Select Weather Scenario:", ("Clear Sunny Day", "Intermittent Clouds", "Heavy Monsoon Day"))
initial_storage_percent = st.sidebar.slider("Set Initial Reservoir Level (%):", min_value=10, max_value=100, value=75, step=5)
run_button = st.sidebar.button("Calculate Optimal Schedule", type="primary")
st.sidebar.markdown("---")
with st.sidebar.expander("About this Project"):
    st.markdown("""This app is an **AI-powered decision-support tool** for managing the critical nexus of water and electricity.
        - **Solar Forecaster:** A CNN-LSTM model predicts solar output. *(NMAE: 7.78%)*
        - **Price Forecaster:** An XGBoost model predicts Indian electricity prices.
        - **AI Operator:** An optimization algorithm uses these forecasts to create a profit-maximizing, disaster-resilient schedule for a hydropower dam.""")

st.title("🌊☀️ The AI Hydro-Solar Nexus Arbitrageur")
st.markdown("An AI tool to co-optimize water & electricity for a sustainable and resilient grid.")
st.markdown("---")

if run_button and price_model is not None and df_master is not None:
    # --- Step 1: Simulate Solar Forecast ---
    if weather_scenario == "Clear Sunny Day": solar_kw = 80000*(np.sin(np.linspace(0,np.pi,24)-.2)**2)
    elif weather_scenario == "Intermittent Clouds": solar_kw = 80000*(np.sin(np.linspace(0,np.pi,24)-.2)**2)*np.random.uniform(.5,1,24)
    else: solar_kw = 15000*(np.sin(np.linspace(0,np.pi,24)-.2)**2)
    solar_mwh = np.clip(solar_kw/1000, 0, None)
    
    last_day_features = df_master[price_model.feature_names_in_].tail(24)
    price_forecast = price_model.predict(last_day_features)
    
    max_storage = df_master['storage_mcm'].max()
    initial_storage = (initial_storage_percent / 100) * max_storage
    actions, hydro_mwh, revenue, storage_trace = get_optimal_schedule(solar_mwh, price_forecast, initial_storage, max_storage, .9, .3, max_storage*.05, 150)
    
    st.header("Optimal 24-Hour Schedule & Results")
    tab1, tab2, tab3 = st.tabs(["📊 Visual Dashboard", "📈 Financial Breakdown", "📋 Action Plan"])

    with tab1:
        st.subheader("Visual Dashboard")
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        hours = np.arange(24)
        
        # ### <<< THE DEFINITIVE FIX: ASSIGN ALL PLOTTING RETURNS TO '_' >>> ###
        _ = axes[0].plot(hours, price_forecast, label='Forecasted Price (Rs/MWh)', color='red', marker='o')
        _ = axes[0].plot(hours, solar_mwh, label='Simulated Solar (MWh)', color='orange', marker='s')
        _ = axes[0].set_title('Forecasted Market & Weather Conditions', fontsize=16)
        _ = axes[0].set_ylabel('Price / Power')
        _ = axes[0].legend()
        _ = axes[0].grid(True)
        
        ax1b = axes[1].twinx()
        _ = axes[1].bar(hours, hydro_mwh, label='Hydropower Schedule (MWh)', color='dodgerblue', alpha=0.7)
        _ = ax1b.plot(np.arange(25), storage_trace / max_storage * 100, label='Reservoir Level (%)', color='green', linestyle='--')
        _ = axes[1].set_title('Prescriptive AI Hydropower & Storage Schedule', fontsize=16)
        _ = axes[1].set_ylabel('Generation (MWh)')
        _ = ax1b.set_ylabel('Reservoir Level (%)')
        _ = ax1b.set_ylim(0, 110)
        _ = axes[1].legend(loc='upper left')
        _ = ax1b.legend(loc='upper right')
        _ = axes[1].grid(True)
        _ = axes[1].set_xlabel('Hour of the Day')
        _ = axes[1].set_xticks(hours)
        
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Financial & Resource Breakdown")
        st.metric(label="Total Projected Hydropower Revenue", value=f"₹ {revenue:,.0f}")
        breakdown_df = pd.DataFrame({'Hour': hours, 'Price (Rs/MWh)': price_forecast.round(0), 'Solar (MWh)': solar_mwh.round(2),
                                     'Hydro (MWh)': hydro_mwh.round(2), 'Revenue (₹)': (hydro_mwh * price_forecast).round(0),
                                     'Reservoir Level (%)': (storage_trace[1:]/max_storage*100).round(1)})
        st.dataframe(breakdown_df)

    with tab3:
        st.subheader("Operational Action Plan")
        action_df = pd.DataFrame({'Hour': hours, 'Prescribed Action': actions})
        st.dataframe(action_df)

else:
    st.info("Set your parameters in the sidebar on the left and click 'Calculate Optimal Schedule' to see the AI in action.")
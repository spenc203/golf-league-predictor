import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# --- 1. CONFIGURATION & Setup ---
st.set_page_config(layout="wide")
st.sidebar.title("Welcome to the Golf Dashboard!")
st.title("⛳ Advanced Golf League Data Dashboard")

# Define file paths
model_dir = os.path.dirname(__file__)
linear_model_path = os.path.join(model_dir, 'linear_model.pkl')
logistic_model_path = os.path.join(model_dir, 'logistic_model.pkl')
historical_data_path = os.path.join(model_dir, 'historical_data.csv')

# --- LOAD ASSETS (Models and Data) ---
@st.cache_resource
def load_assets():
    """Loads models and static data once."""
    try:
        linear_model = joblib.load(linear_model_path)
        logistic_model = joblib.load(logistic_model_path)
        historical_data = pd.read_csv(historical_data_path)
        feature_names = linear_model.feature_names_in_.tolist()
        
        # Prepare list of player names based on model features
        PLAYER_FEATURE_NAMES = [name for name in feature_names if name.startswith('PlayerName_')]
        player_names = sorted([name.replace("PlayerName_", "") for name in PLAYER_FEATURE_NAMES])

        RMSE_VALUE = 2.93
        
        return linear_model, logistic_model, historical_data, feature_names, player_names, RMSE_VALUE
    except Exception as e:
        st.error(f"Error loading models or data: {e}. Please ensure all .pkl and .csv files are present.")
        st.stop()

linear_model, logistic_model, historical_data, feature_names, player_names, RMSE_VALUE = load_assets()


# --- 2. PREDICTION FUNCTION (NO CACHE) ---
def get_predictions(df):
    """Calculates predictions and probability using static models."""
    global linear_model, logistic_model
    
    # 1. LINEAR REGRESSION PREDICTION
    predicted_score = linear_model.predict(df)[0]
    
    # 2. LOGISTIC REGRESSION PROBABILITY
    probability_to_win = logistic_model.predict_proba(df)[0][1] * 100 
    
    return predicted_score, probability_to_win


# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Round Inputs")
    
    selected_player = st.selectbox("1. Select Player", player_names)
    
    current_handicap = st.number_input("2. Current Handicap (Numeric)", min_value=0.0, max_value=30.0, value=12.0, step=0.1)
    
    previous_score = st.number_input("3. Previous Round Over Par Score", min_value=-10.0, max_value=20.0, value=5.0, step=0.1)
    
    is_front_nine = st.radio("4. Course Side", options=["Front Nine", "Back Nine"])


# --- 3. FEATURE ENGINEERING (REQUIRED FOR PREDICTION) ---
final_input_data = {}

for col in feature_names:
    final_input_data[col] = [0] 

# Map current inputs to ORIGINAL feature names: HandicapPre, Lag_OverPar
final_input_data['HandicapPre'] = [current_handicap]
final_input_data['Lag_OverPar'] = [previous_score]

# Map Course Side to ORIGINAL feature names: Links_Front Nine
if is_front_nine == "Front Nine":
    if 'Links_Front Nine' in final_input_data:
         final_input_data['Links_Front Nine'] = [1]

player_skill_feature = f'PlayerName_{selected_player}'
final_input_data[player_skill_feature] = [1]
    
final_input_df = pd.DataFrame(final_input_data, columns=feature_names)

# --- RUN PREDICTIONS ---
predicted_score, probability_to_win = get_predictions(final_input_df)


# --- 4. DISPLAY METRICS ---
st.header("🔮 Predicted Performance")
col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Score (Strokes Over Par)", f"{predicted_score:.2f}")

with col2:
    st.metric("**Probability to Score Better Than Average**", f"{probability_to_win:.1f}%")

st.caption(f"Average Prediction Error (RMSE): ±{RMSE_VALUE:.2f} strokes")
st.markdown("---")

# --- 5. EXPLANATION AND CHARTS (HISTORICAL ONLY) ---
st.header("📊 Contextual Analysis")

# Use a single container instead of tabs since we removed the second chart
st.subheader(f"Historical Trend for {selected_player}")
    
player_history = historical_data[historical_data['PlayerName'] == selected_player].copy()

if not player_history.empty:
    player_history['RoundNumber'] = range(1, len(player_history) + 1)
    
    fig = px.scatter(
        player_history, 
        x='RoundNumber', 
        y='OverPar', 
        color='PlayerName', 
        trendline='ols',
        title='Score Over Time (Trendline: OLS Regression)',
        labels={'OverPar': 'Score (Strokes Over Par)', 'RoundNumber': 'Round Number'}
    )
    # Add the current prediction as a large red mark (using the next round number)
    next_round = player_history['RoundNumber'].max() + 1 if not player_history.empty else 1
    fig.add_scatter(
        x=[next_round], 
        y=[predicted_score], 
        mode='markers', 
        marker=dict(size=12, color='red', symbol='star'), 
        name='Current Prediction'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No historical data found for this player.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide")
st.sidebar.title("Welcome to the Golf Dashboard!")
st.title("⛳ Advanced Golf League Data Dashboard")

# Define file paths
model_dir = os.path.dirname(__file__)
linear_model_path = os.path.join(model_dir, 'linear_model.pkl')
logistic_model_path = os.path.join(model_dir, 'logistic_model.pkl')
historical_data_path = os.path.join(model_dir, 'historical_data.csv')

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        linear_model = joblib.load(linear_model_path)
        logistic_model = joblib.load(logistic_model_path)
        historical_data = pd.read_csv(historical_data_path)
        # Create the input feature list (all features except the one being predicted)
        feature_names = linear_model.feature_names_in_.tolist()
        return linear_model, logistic_model, historical_data, feature_names
    except Exception as e:
        st.error(f"Error loading models or data: {e}. Please ensure all .pkl and .csv files are present.")
        st.stop()

linear_model, logistic_model, historical_data, feature_names = load_assets()

# Get the unique player names for the dropdown
player_names = sorted(historical_data['PlayerName'].unique().tolist())

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Round Inputs")
    
    selected_player = st.selectbox("1. Select Player", player_names)
    
    # Filter features based on the selected player (excluding the player's own dummy var)
    filtered_features = [f for f in feature_names if f != f'PlayerName_{selected_player}']
    
    current_handicap = st.number_input("2. Current Handicap (Numeric)", min_value=0.0, max_value=30.0, value=12.0, step=0.1)
    
    previous_score = st.number_input("3. Previous Round Over Par Score", min_value=-10.0, max_value=20.0, value=5.0, step=0.1)
    
    is_front_nine = st.radio("4. Course Side", options=["Front Nine", "Back Nine"])
    
    # --- PREPARE INPUT DATAFRAME ---
    input_data = {
        'Handicap': [current_handicap],
        'PrevRoundScore': [previous_score],
        'CourseSide_Front Nine': [1 if is_front_nine == "Front Nine" else 0],
        'CourseSide_Back Nine': [1 if is_front_nine == "Back Nine" else 0],
    }

    # Dynamically add the selected player's dummy variable (set to 1)
    for player in player_names:
        col_name = f'PlayerName_{player}'
        input_data[col_name] = [1 if player == selected_player else 0]

    # Create the DataFrame and ensure column order matches training data
    input_df = pd.DataFrame(input_data)
    
    # Ensure all necessary feature columns (including the unselected players) are present and in order
    final_input_df = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in input_df.columns:
            final_input_df[col] = input_df[col]
        else:
            final_input_df[col] = 0 # Default other dummy variables to 0

# --- 1. LINEAR REGRESSION PREDICTION ---
st.header("🔮 Predicted Performance")
col1, col2 = st.columns(2)

# Prediction Logic
predicted_score = linear_model.predict(final_input_df)[0]

with col1:
    st.metric("Predicted Score (Strokes Over Par)", f"{predicted_score:.2f}")

# --- 2. LOGISTIC REGRESSION PROBABILITY ---

# Probability Logic
# Predicts the probability of the POSITIVE class (1: Better than Average)
probability_to_win = logistic_model.predict_proba(final_input_df)[0][1] * 100 

with col2:
    # **CORRECTED LABEL:** This is the key fix for the ambiguity you found!
    st.metric("**Probability to Score Better Than Average**", f"{probability_to_win:.1f}%")

st.markdown("---")

# --- 3. EXPLANATION AND CHARTS ---
st.header("📊 Contextual Analysis")

tab1, tab2 = st.tabs(["Player Historical Trend", "Model Feature Impact"])

# --- TAB 1: HISTORICAL TREND ---
with tab1:
    st.subheader(f"{selected_player}'s Scoring History")
    player_history = historical_data[historical_data['PlayerName'] == selected_player].copy()
    
    if not player_history.empty:
        # Create a sequence of rounds for the x-axis
        player_history['RoundNumber'] = player_history.index + 1
        
        fig = px.scatter(
            player_history, 
            x='RoundNumber', 
            y='OverPar', 
            color='PlayerName', 
            trendline='ols',
            title=f'{selected_player}: Score Over Time (Trendline: OLS Regression)',
            labels={'OverPar': 'Score (Strokes Over Par)', 'RoundNumber': 'Round Number'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data found for this player.")

# --- TAB 2: MODEL COEFFICIENTS ---
with tab2:
    st.subheader("Model Weights: How Inputs Influence
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
        # Note: feature_names_in_ holds the column order from training
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
    
    current_handicap = st.number_input("2. Current Handicap (Numeric)", min_value=0.0, max_value=30.0, value=12.0, step=0.1)
    
    previous_score = st.number_input("3. Previous Round Over Par Score", min_value=-10.0, max_value=20.0, value=5.0, step=0.1)
    
    is_front_nine = st.radio("4. Course Side", options=["Front Nine", "Back Nine"])
    
    # --- PREPARE INPUT DATAFRAME (CRITICAL FIX FOR NaN/ValueError) ---
    
    # Dictionary to hold the user inputs
    input_data_user = {
        'Handicap': [current_handicap],
        'PrevRoundScore': [previous_score],
        # Only one of these will be 1, the other is handled below by default=0
        'CourseSide_Front Nine': [1 if is_front_nine == "Front Nine" else 0],
        'CourseSide_Back Nine': [1 if is_front_nine == "Back Nine" else 0],
    }

    # Dynamically add the selected player's dummy variable (set to 1)
    player_skill_feature = f'PlayerName_{selected_player}'
    input_data_user[player_skill_feature] = [1]
    
    # Final structured data dictionary (ensures all model features are present)
    final_input_data = {}
    
    for col in feature_names:
        if col in input_data_user:
            # Transfer the user input/1
            final_input_data[col] = input_data_user[col]
        else:
            # Set all other dummy variables (unselected players, course side not chosen) to 0
            final_input_data[col] = [0]
            
    # Create the final DataFrame, guaranteeing correct
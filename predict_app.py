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
    
    # --- PREPARE INPUT DATAFRAME (Robust creation) ---
    
    # Dictionary to hold the features required by the model
    final_input_data = {}

    # 1. Initialize all features required by the model to [0]
    for col in feature_names:
        final_input_data[col] = [0] 

    # 2. Add the user-defined numerical values
    final_input_data['Handicap'] = [current_handicap]
    final_input_data['PrevRoundScore'] = [previous_score]

    # 3. Set the specific dummy variables to [1]
    if is_front_nine == "Front Nine":
        final_input_data['CourseSide_Front Nine'] = [1]
    else:
        final_input_data['CourseSide_Back Nine'] = [1] 

    # Set the selected player's dummy variable to [1]
    player_skill_feature = f'PlayerName_{selected_player}'
    final_input_data[player_skill_feature] = [1]
        
    # Create the final DataFrame, guaranteeing correct columns and a single row
    final_input_df = pd.DataFrame(final_input_data, columns=feature_names)


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
    # CORRECTED LABEL
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
        # CRITICAL FIX for ValueError
        player_history['RoundNumber'] = range(1, len(player_history) + 1)
        
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
    st.subheader("Model Weights: How Inputs Influence Prediction")
    
    # Extract Coefficients 
    coef_df = pd.DataFrame({
        'Feature': linear_model.feature_names_in_,
        'Coefficient': linear_model.coef_[0]
    })
    
    # Highlight the selected player's skill coefficient
    player_skill_feature = f'PlayerName_{selected_player}'
    
    # CRITICAL FIX: Define ALL features to display for a comprehensive portfolio view.
    features_to_display = [
        'Handicap', 
        'PrevRoundScore', 
        'CourseSide_Front Nine', 
        'CourseSide_Back Nine',
        player_skill_feature  
    ]
    
    # Filter the DataFrame to only show the chosen features
    display_coef_df = coef_df[coef_df['Feature'].isin(features_to_display)].copy()
    
    # Rename features for better display
    display_coef_df['Feature'] = display_coef_df['Feature'].replace({
        'Handicap': 'Current Handicap',
        'PrevRoundScore': 'Previous Score Momentum',
        'CourseSide_Front Nine': 'Front Nine Bias',
        'CourseSide_Back Nine': 'Back Nine Bias',
        player_skill_feature: 'Player Skill Factor' 
    })
    
    # Create Bar Chart
    fig_coef = px.bar(
        display_coef_df, 
        x='Coefficient', 
        y='Feature', 
        orientation='h',
        title='Impact of Key Factors on Predicted Score',
        labels={'Coefficient': 'Impact on Predicted Score (Lower is Better)'}
    )
    # Highlight colors: Green for negative (good), Red for positive (bad)
    fig_coef.update_traces(marker_color=['red' if c > 0 else 'green' for c in display_coef_df['Coefficient']])

    st.plotly_chart(fig_coef, use_container_width=True)
    
    st.markdown(
        """
        *A **Negative Coefficient (Green)** means that factor (e.g., lower Handicap) drives the predicted score **DOWN** (better).* *A **Positive Coefficient (Red)** means that factor (e.g., higher Previous Score) drives the predicted score **UP** (worse).*
        """
    )
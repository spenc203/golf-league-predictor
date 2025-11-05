import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# --- Configuration & Data Loading ---
st.set_page_config(layout="wide") 

try:
    # Load Regression Model (Predicts the Score)
    model = joblib.load('linear_model.pkl')
    # Load Classification Model (Predicts the Probability)
    log_model = joblib.load('logistic_model.pkl')
    # Load Historical Data for charting
    historical_data = pd.read_csv('historical_data.csv')
except FileNotFoundError:
    st.error("One or more required files (models or historical_data.csv) not found. Please run the prerequisite step in Jupyter.")
    st.stop()

# Use the features the model was trained on
MODEL_FEATURE_NAMES = model.feature_names_in_
PLAYER_FEATURE_NAMES = [name for name in MODEL_FEATURE_NAMES if name.startswith('PlayerName_')]
SIMPLE_PLAYER_NAMES = [name.replace("PlayerName_", "") for name in PLAYER_FEATURE_NAMES]
SIMPLE_PLAYER_NAMES.sort()

RMSE_VALUE = 2.93 

# --- Function to Create Feature Contribution Chart ---
def get_feature_contributions(coefs, features):
    # Combine feature names and coefficients
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefs})
    
    # Identify and categorize key features
    contribution_list = []
    
    # Handicap and Previous Score
    if 'HandicapPre' in features:
        hcap_coef = coef_df[coef_df['Feature'] == 'HandicapPre']['Coefficient'].iloc[0]
        contribution_list.append({'Factor': 'Handicap Adjustment', 'Impact': hcap_coef})
    if 'Lag_OverPar' in features:
        lag_coef = coef_df[coef_df['Feature'] == 'Lag_OverPar']['Coefficient'].iloc[0]
        contribution_list.append({'Factor': 'Previous Score Momentum', 'Impact': lag_coef})
    if 'Links_Front Nine' in features:
        links_coef = coef_df[coef_df['Feature'] == 'Links_Front Nine']['Coefficient'].iloc[0]
        contribution_list.append({'Factor': 'Front Nine Course Bias', 'Impact': links_coef})
        
    # Player Baseline (The dropped dummy is 0)
    baseline_player_name = [name for name in SIMPLE_PLAYER_NAMES if f'PlayerName_{name}' not in PLAYER_FEATURE_NAMES]
    if baseline_player_name:
         contribution_list.append({'Factor': f'Baseline Player: {baseline_player_name[0]}', 'Impact': 0.0})

    # Add Player specific coefficient
    player_col_name_full = f'PlayerName_{st.session_state.get("selected_player")}'
    if player_col_name_full in features:
        player_coef = coef_df[coef_df['Feature'] == player_col_name_full]['Coefficient'].iloc[0]
        contribution_list.append({'Factor': f'Player Skill: {st.session_state.get("selected_player")}', 'Impact': player_coef})

    contribution_df = pd.DataFrame(contribution_list)
    return contribution_df.sort_values(by='Impact', ascending=False)

# --- Streamlit App Layout ---
st.title("⛳ Golf Score Predictor: 2025 League")
st.markdown("### Advanced Prediction and Analysis Dashboard")

col_input, col_metrics = st.columns([1, 1])

# --- Input Panel ---
with col_input:
    st.header("Input Parameters")
    
    selected_player = st.selectbox("Select Player:", SIMPLE_PLAYER_NAMES, key='selected_player')
    
    st.markdown("---")
    
    handicap = st.slider(
        "Handicap (Pre-Round):", 
        min_value=0.0, max_value=30.0, value=15.0, step=1.0, format="%.0f"
    )
    course_side = st.radio("Course Side:", ('Front Nine', 'Back Nine'))
    lag_overpar = st.slider(
        "Previous Round OverPar Score:", 
        min_value=-5.0, max_value=20.0, value=5.0, step=1.0, format="%.0f"
    )

# --- Feature Engineering ---
input_data = pd.DataFrame(0, index=[0], columns=MODEL_FEATURE_NAMES)
if 'HandicapPre' in input_data.columns: input_data['HandicapPre'] = handicap
if 'Lag_OverPar' in input_data.columns: input_data['Lag_OverPar'] = lag_overpar
if course_side == 'Front Nine':
    if 'Links_Front Nine' in input_data.columns: input_data['Links_Front Nine'] = 1

player_col_name_full = f'PlayerName_{selected_player}'
if player_col_name_full in input_data.columns:
    input_data[player_col_name_full] = 1


# --- Metrics Panel ---
with col_metrics:
    st.header("Prediction Results")
    
    # 1. Regression Prediction (The Score)
    predicted_overpar = model.predict(input_data)[0]
    
    # 2. Classification Prediction (The Probability)
    # The output is [Prob_0, Prob_1]. We want Prob_1 (Beating Handicap).
    prob_beat_hcap = log_model.predict_proba(input_data)[0][1] * 100 

    st.metric(
        label=f"Predicted OverPar Score:",
        value=f"{predicted_overpar:.2f} Strokes",
        help="Linear Regression Model Output"
    )
    
    st.metric(
        label=f"Probability to Beat Handicap:",
        value=f"{prob_beat_hcap:.1f}%",
        help="Logistic Regression Model Output: Chance of Net Score < 0"
    )
    st.caption(f"Average Prediction Error (RMSE): ±{RMSE_VALUE:.2f} strokes")


# --- Charts Section ---
st.markdown("---")
st.header("Analysis and Context")

col_hist, col_contrib = st.columns(2)

# 1. Historical Chart (Visualizing the "Why")
with col_hist:
    st.subheader(f"Historical Trend for {selected_player}")
    
    player_history = historical_data[historical_data['PlayerName'] == selected_player]
    
    # Plotly Scatter Plot
    fig = px.scatter(
        player_history, 
        x='RoundNumber', 
        y='OverPar', 
        trendline='ols', # Add a linear trend line
        title='Score vs. Time (OverPar)',
        labels={'OverPar': 'Score (Over Par)', 'RoundNumber': 'Round Number'}
    )
    
    # Add the current prediction as a large red mark (using the next round number)
    next_round = player_history['RoundNumber'].max() + 1 if not player_history.empty else 1
    fig.add_scatter(
        x=[next_round], 
        y=[predicted_overpar], 
        mode='markers', 
        marker=dict(size=12, color='red', symbol='star'), 
        name='Current Prediction'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 2. Feature Contributions Chart (Visualizing the "How")
with col_contrib:
    st.subheader("Model Feature Impact (Strokes Added/Subtracted)")

    # Get the coefficients relative to the current player selection
    contrib_df = get_feature_contributions(model.coef_, MODEL_FEATURE_NAMES)
    
    fig_contrib = px.bar(
        contrib_df, 
        y='Factor', 
        x='Impact', 
        orientation='h',
        color='Impact',
        color_continuous_scale=px.colors.diverging.RdBu,
        labels={'Impact': 'Impact on Predicted OverPar Score (Strokes)'}
    )
    
    # Add a vertical line at x=0 for clarity
    fig_contrib.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
    
    st.plotly_chart(fig_contrib, use_container_width=True)
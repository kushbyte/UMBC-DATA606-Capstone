import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import altair as alt
import os

# --- App Constants ---
# We will now define the file name, and find its *full path* inside the app
CSV_FILE_NAME = 'nba_all_players_game_logs_2023-24.csv'
CUTOFF_DATE = '2024-03-01'
ALL_FEATURE_NAMES = [
    'PTS_roll_5', 'MIN_roll_5', 'FGA_roll_5', 'FTA_roll_5', 
    'AST_roll_5', 'REB_roll_5', 'FG3A_roll_5', 'TOV_roll_5',
    'PTS_std_5', 'MIN_std_5', 'USG_roll_5', 'PACE_roll_5',
    'OPP_DEF_STR_roll_5', 'TEAM_PTS_roll_5', 'OPP_PACE_roll_5',
    'HOME', 'Days_Rest'
]

# --- Master Pipeline Function ---

@st.cache_data(show_spinner="Running analysis for the first time...")
def load_and_run_pipeline(file_path):
    """
    Loads data, engineers all features, trains the model,
    and returns all artifacts needed for the dashboard.
    This entire function is cached.
    """
    
    # 1. LOAD DATA
    if not os.path.exists(file_path):
        # This error will be displayed in the Streamlit app
        return f"Error: File not found. The script looked for it at this exact path: `{file_path}`. Please ensure the file is there."
    
    df = pd.read_csv(file_path)

    # 2. FEATURE ENGINEERING
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['TEAM'] = df['MATCHUP'].apply(lambda x: x.split()[0])
    df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.split()[-1])
    df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    df = df.sort_values(['Player_ID', 'GAME_DATE'])
    df['Days_Rest'] = df.groupby('Player_ID')['GAME_DATE'].diff().dt.days
    df['Days_Rest'] = df['Days_Rest'].fillna(7)
    df['USG_proxy'] = df['FGA'] + df['FTA'] + df['TOV']
    df['PACE_proxy'] = df['FGA'] + df['AST'] + df['TOV']

    # Player Rolling Features
    player_roll_cols = {
        'PTS': ['mean', 'std'], 'MIN': ['mean', 'std'], 'FGA': ['mean'],
        'FTA': ['mean'], 'AST': ['mean'], 'REB': ['mean'],
        'FG3A': ['mean'], 'TOV': ['mean'], 'USG_proxy': ['mean'], 'PACE_proxy': ['mean']
    }
    grouped = df.groupby('Player_ID')
    for col, funcs in player_roll_cols.items():
        for func in funcs:
            new_col_name = f'{col}_roll_5' if func == 'mean' else f'{col}_std_5'
            # Calculate rolling feature
            rolling_stat = grouped[col].rolling(window=5, min_periods=1).agg(func)
            # Shift to prevent data leakage and add to dataframe
            df[new_col_name] = rolling_stat.shift(1).reset_index(level=0, drop=True)
    
    df['PTS_std_5'] = df['PTS_std_5'].fillna(0)
    df['MIN_std_5'] = df['MIN_std_5'].fillna(0)
    df.rename(columns={'USG_proxy_roll_5': 'USG_roll_5', 'PACE_proxy_roll_5': 'PACE_roll_5'}, inplace=True)

    # Team Rolling Features
    game_id_col = 'Game_ID' if 'Game_ID' in df.columns else 'GAME_ID'
    
    team_offense = df.groupby([game_id_col, 'TEAM', 'GAME_DATE'])['PTS'].sum().reset_index()
    team_offense = team_offense.sort_values(['TEAM', 'GAME_DATE'])
    team_offense['TEAM_PTS_roll_5'] = team_offense.groupby('TEAM')['PTS'].rolling(5, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)

    team_defense = df.groupby([game_id_col, 'OPPONENT', 'GAME_DATE'])['PTS'].sum().reset_index()
    team_defense = team_defense.sort_values(['OPPONENT', 'GAME_DATE'])
    team_defense['OPP_DEF_STR_roll_5'] = team_defense.groupby('OPPONENT')['PTS'].rolling(5, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)

    team_pace = df.groupby([game_id_col, 'OPPONENT', 'GAME_DATE'])['PACE_proxy'].sum().reset_index()
    team_pace = team_pace.sort_values(['OPPONENT', 'GAME_DATE'])
    team_pace['OPP_PACE_roll_5'] = team_pace.groupby('OPPONENT')['PACE_proxy'].rolling(5, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)

    # Merge Team Features
    df = df.merge(team_offense[['TEAM', 'GAME_DATE', 'TEAM_PTS_roll_5']], on=['TEAM', 'GAME_DATE'], how='left')
    df = df.merge(team_defense[['OPPONENT', 'GAME_DATE', 'OPP_DEF_STR_roll_5']], on=['OPPONENT', 'GAME_DATE'], how='left')
    df = df.merge(team_pace[['OPPONENT', 'GAME_DATE', 'OPP_PACE_roll_5']], on=['OPPONENT', 'GAME_DATE'], how='left')

    # Final Cleanup
    league_avg_pts = team_offense['PTS'].mean()
    league_avg_pace = team_pace['PACE_proxy'].mean()
    df['TEAM_PTS_roll_5'] = df['TEAM_PTS_roll_5'].fillna(league_avg_pts)
    # Fix potential typo from previous generation
    if 'OPP_DEF_STR_roll_5' in df.columns:
        df['OPP_DEF_STR_roll_5'] = df['OPP_DEF_STR_roll_5'].fillna(league_avg_pts)
    df['OPP_PACE_roll_5'] = df['OPP_PACE_roll_5'].fillna(league_avg_pace)
    df = df.drop(columns=['USG_proxy', 'PACE_proxy'])
    
    # 3. MODEL TRAINING & PREDICTION
    # We create df_model from the *full* dataset before splitting
    df_model = df.dropna(subset=['PTS_roll_5']).copy()
    X = df_model[ALL_FEATURE_NAMES]
    y = df_model['PTS']
    
    train_mask = df_model['GAME_DATE'] < CUTOFF_DATE
    test_mask = df_model['GAME_DATE'] >= CUTOFF_DATE
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Final NaN check
    train_non_nan_indices = X_train.dropna().index
    X_train_clean = X_train.loc[train_non_nan_indices]
    y_train_clean = y_train.loc[train_non_nan_indices]

    test_non_nan_indices = X_test.dropna().index
    X_test_clean = X_test.loc[test_non_nan_indices]
    y_test_clean = y_test.loc[test_non_nan_indices]
    
    # This is the dataframe with test set predictions
    df_model_test = df_model.loc[test_non_nan_indices].copy()
    
    model = LinearRegression()
    model.fit(X_train_clean, y_train_clean)
    
    predictions = model.predict(X_test_clean)
    lr_mae = mean_absolute_error(y_test_clean, predictions)
    
    baseline_predictions = X_test_clean['PTS_roll_5']
    baseline_mae = mean_absolute_error(y_test_clean, baseline_predictions)
    
    # --- 4. CREATE FINAL ARTIFACTS ---
    df_model_test.rename(columns={'PTS': 'Actual_PTS'}, inplace=True)
    df_model_test['Predicted_PTS'] = predictions.round(1)
    df_model_test['Difference'] = df_model_test['Actual_PTS'] - df_model_test['Predicted_PTS']
    df_model_test['Difference_Abs'] = df_model_test['Difference'].abs()
    
    coef_df = pd.DataFrame({'Feature': ALL_FEATURE_NAMES, 'Coefficient': model.coef_})
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    
    # *** NEW RETURN ***
    # We now return the full df_model along with the other artifacts
    return lr_mae, baseline_mae, df_model_test, coef_df, df_model
    
    
    # --- Main Application UI ---

def main():
    st.set_page_config(layout="wide")
    
    # --- NEW ROBUST PATH LOGIC ---
    # Get the absolute path to the directory containing this script
    try:
        # __file__ is a special variable that holds the path to the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for environments where __file__ is not defined (like some notebooks)
        script_dir = os.getcwd() 
        
    # Define the full, absolute path to the CSV file
    csv_file_path = os.path.join(script_dir, CSV_FILE_NAME)
    
    st.title("🏀 NBA Player Points Prediction Dashboard")
    st.markdown(f"""
    This dashboard shows the results of the Linear Regression model built in the project.
    - **Data Source:** `{CSV_FILE_NAME}` (Loaded from script directory)
    - **Test Set:** All games from {CUTOFF_DATE} to the end of the season.
    - **Model:** Linear Regression
    - **Key Finding:** The model's primary driver is **`USG_roll_5`** (a proxy for player opportunity), which is a stronger predictor than past points (`PTS_roll_5`).
    """)

    # --- Load all data from the cached function ---
    artifacts = load_and_run_pipeline(csv_file_path) # Pass the new full path
    
    # Check if the pipeline returned an error string
    if isinstance(artifacts, str):
        st.error(artifacts) # Display the file not found error
        st.stop() # Stop the app

    # *** NEW UNPACK ***
    lr_mae, baseline_mae, comp_df, coef_df, df_model = artifacts

    # --- Section 1: Top-Line Performance ---
    st.header(f"Model Performance (Test Set: {CUTOFF_DATE} onwards)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model MAE", f"{lr_mae:.2f} points", help="The model's average prediction error.")
    col2.metric("Baseline MAE", f"{baseline_mae:.2f} points", help="The average error if we just guessed the player's 5-game average.")
    col3.metric("Improvement", f"{baseline_mae - lr_mae:+.2f} points", help=f"The model is {baseline_mae - lr_mae:.2f} points more accurate than the simple baseline.")

    # --- Section 2: Prediction Explorer (UPDATED) ---
    st.header("Prediction Explorer")
    
    # The player list now comes from the *full* df_model
    player_list = sorted(df_model['PLAYER_NAME'].unique())
    selected_player = st.selectbox("Select a Player to analyze:", player_list)
    
    # --- NEW: Full Season Stats Section ---
    st.subheader(f"Full Season Stats for {selected_player}")
    # Get all games for the selected player from the full model dataframe
    player_full_season_df = df_model[df_model['PLAYER_NAME'] == selected_player].sort_values('GAME_DATE')
    
    # Show season averages
    avg_pts = player_full_season_df['PTS'].mean()
    avg_reb = player_full_season_df['REB'].mean()
    avg_ast = player_full_season_df['AST'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Season PPG", f"{avg_pts:.1f}")
    col2.metric("Season RPG", f"{avg_reb:.1f}")
    col3.metric("Season APG", f"{avg_ast:.1f}")

    with st.expander("Show Full Season Game Log (Raw Stats)"):
        st.dataframe(player_full_season_df[['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST', 'FGA', 'USG_roll_5']].style.format({
            "USG_roll_5": "{:.1f}"
        }))

    # --- EXISTING: Test Set Predictions Section ---
    st.subheader(f"Test Set Predictions for {selected_player}")
    
    # Filter the *test set* dataframe (comp_df)
    player_test_df = comp_df[comp_df['PLAYER_NAME'] == selected_player].sort_values('GAME_DATE')
    
    # Check if this player has games in the test set
    if player_test_df.empty:
        st.info(f"{selected_player} has no games in the test set (from {CUTOFF_DATE} onwards).")
    else:
        # Metric for this player's test set games
        player_mae = player_test_df['Difference_Abs'].mean()
        st.metric(f"Avg. Error for {selected_player} (in Test Set)", f"{player_mae:.2f} points")
        
        # Chart
        st.markdown("##### Actual vs. Predicted Points Over Time (Test Set)")
        log_df = player_test_df[['GAME_DATE', 'Actual_PTS', 'Predicted_PTS']].melt('GAME_DATE', var_name='Metric', value_name='Points')
        
        chart = alt.Chart(log_df).mark_line(point=True).encode(
            x=alt.X('GAME_DATE', title='Game Date'),
            y=alt.Y('Points', title='Points'),
            color='Metric',
            tooltip=['GAME_DATE', 'Metric', 'Points']
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        # Table
        with st.expander("Show Raw Prediction Data for " + selected_player):
            st.dataframe(player_test_df[['GAME_DATE', 'MATCHUP', 'Actual_PTS', 'Predicted_PTS', 'Difference']].style.format({
                "Predicted_PTS": "{:.1f}",
                "Difference": "{:+.1f}"
            }))

    # --- Section 3: Model Insights ---
    st.header("Model Insights: What Drives the Predictions?")
    st.markdown("This chart shows the model's learned coefficients. A high *absolute* value means the feature is very. **`USG_roll_5`** is the most important feature.")
    
    importance_chart = alt.Chart(coef_df).mark_bar().encode(
        x=alt.X('Abs_Coefficient', title='Importance (Absolute Coefficient)'),
        y=alt.Y('Feature', sort='-x'),
        color=alt.condition(
            alt.datum.Coefficient > 0,
            alt.value("#008000"),  # Green for positive
            alt.value("#FF0000")   # Red for negative
        ),
        tooltip=['Feature', alt.Tooltip('Coefficient', format='.2f')]
    ).properties(
        title="Model Feature Importance"
    ).interactive()
    
    st.altair_chart(importance_chart, use_container_width=True)

    # --- Section 4: Notable Predictions ---
    st.header("Notable Predictions (Entire Test Set)")
    col1, col2 = st.columns(2)
    
    # *** THIS IS THE FIX ***
    # Removed the leading space from 'Predicted_PTS'
    display_cols = ['PLAYER_NAME', 'MATCHUP', 'Actual_PTS', 'Predicted_PTS', 'Difference']
    
    with col1:
        st.subheader("✅ 5 Best Predictions")
        st.dataframe(
            comp_df.sort_values('Difference_Abs').head(5)[display_cols].style.format({
                "Predicted_PTS": "{:.1f}", "Difference": "{:+.1f}"
            })
        )
    
    with col2:
        st.subheader("❌ 5 Worst Predictions")
        st.dataframe(
            comp_df.sort_values('Difference_Abs', ascending=False).head(5)[display_cols].style.format({
                "Predicted_PTS": "{:.1f}", "Difference": "{:+.1f}"
            })
        )
if __name__ == "__main__":
    main()
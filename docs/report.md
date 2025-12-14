# 1. Title and Author

# 📊 Predicting NBA Player Performance (Points) 2023 - 24 Season

* **Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang**
* **Author Name:** Kushal Senapathi
* **Semester:** Fall 2025
---

## 🔗 Project Resources & Author Links

* **YouTube Video Presentation:** [https://youtu.be/svWFzMKRZBE?si=HBbTt-jl0nxeAZVZ]
* **PPT Presentation File (in docs):** [https://github.com/kushbyte/UMBC-DATA606-Capstone/blob/main/docs/NBA_ppt.pptx]
* **GitHub Repository:** [https://github.com/kushbyte/UMBC-DATA606-Capstone/tree/main]
* **Author's LinkedIn Profile:** [https://www.linkedin.com/in/kushalsenapathi/]

---

## 2. Background

The project aims to construct a robust machine learning model capable of predicting the number of points an individual NBA player will score in a future game.

* **What is it about?** The primary goal is to determine if historical player statistics, coupled with game and opponent context, can reliably forecast a player’s offensive output.
* **Why does it matter?** Accurate player performance prediction is highly valuable for data-driven applications in professional sports, including coaching strategy, fantasy sports, and real-time betting markets.
* **What are your research questions?** Can a regression model, trained on engineered trend and context features, achieve a significantly lower prediction error (MAE) than a simple historical average (Baseline model)?

---

## 3. Data

### Data Sources and Overview

* **Data Sources:** NBA player game logs, acquired using the `nba-api` Python package.
* **Data Size/Shape:** The initial dataset consists of **23,770 rows** and **28 columns**.
* **Time Period:** The data covers the entirety of the **2023–24 NBA season**.
* **Row Representation:** Each row represents a single **player's performance in one specific game** (a game log).

### Data Dictionary: Initial Raw Data Columns (Before Feature Engineering)

The initial dataset contained standard game log statistics. The columns below represent the core metrics used to build the final features.

| Column Name | Data Type | Definition | Role in Modeling |
| :--- | :--- | :--- | :--- |
| **`PTS`** | Numerical (Integer) | Total points scored in the game. **(TARGET)** | Target Variable |
| `PLAYER_ID` | Numerical (Integer) | Unique identifier for each player. | Identifier |
| `GAME_ID` | Numerical (Integer) | Unique identifier for each game. | Identifier |
| `GAME_DATE` | Date/Time | The date the game was played. | Used for time-series and train/test split. |
| `MIN` | Numerical (Integer) | Minutes played in the game. | Core Feature |
| `FGM` | Numerical (Integer) | Field Goals Made. | Core Feature |
| `FGA` | Numerical (Integer) | Field Goal Attempts. | Core Feature |
| `FTM` | Numerical (Integer) | Free Throws Made. | Core Feature |
| `FTA` | Numerical (Integer) | Free Throw Attempts. | Core Feature |
| `REB` | Numerical (Integer) | Total Rebounds. | Core Feature |
| `AST` | Numerical (Integer) | Assists. | Core Feature |
| `STL` | Numerical (Integer) | Steals. | Core Feature |
| `BLK` | Numerical (Integer) | Blocks. | Core Feature |
| `TOV` | Numerical (Integer) | Turnovers. | Core Feature |
| `PLUS_MINUS` | Numerical (Integer) | Player's plus/minus rating for the game. | Core Feature |
| `WL` | Categorical (String) | Game result (Win or Loss). | Context Feature |
| `TEAM_ID` | Numerical (Integer) | ID of the player's team. | Identifier |
| `OPPONENT_ID` | Numerical (Integer) | ID of the opposing team. | Identifier |
| *Note: Not all 28 raw columns are shown, only the main features used.* | | | |

### Data Dictionary: Final Feature Set (After Feature Engineering)

The final dataset consists of 18 columns. The original player metrics were transformed into rolling averages, which serve as the final predictors.

* **Target/Label Variable:** `PTS`
* **Feature/Predictor Variables:** The remaining **17** variables.

| Column Name | Data Type | Definition | Feature Group |
| :--- | :--- | :--- | :--- |
| **`PTS`** | Numerical (Integer) | Total points scored in the game. **(TARGET/LABEL)** | Target Variable |
| **Player Rolling Averages (5-Game Trend)** | | | |
| `PTS_roll_5` | Numerical (Float) | Player’s average points in the previous 5 games. | Trend/Form |
| `PTS_std_5` | Numerical (Float) | Player’s standard deviation of points in the previous 5 games (volatility). | Trend/Form |
| `MIN_roll_5` | Numerical (Integer) | Player’s average minutes played in the previous 5 games (opportunity). | Usage/Opportunity |
| `MIN_std_5` | Numerical (Float) | Player’s standard deviation of minutes played in the previous 5 games. | Usage/Opportunity |
| `FGA_roll_5` | Numerical (Float) | Player’s average Field Goal Attempts in the previous 5 games (usage). | Usage/Opportunity |
| `FTA_roll_5` | Numerical (Float) | Player’s average Free Throw Attempts in the previous 5 games. | Usage/Opportunity |
| `REB_roll_5` | Numerical (Float) | Player's average Rebounds in the previous 5 games. | Secondary Metric |
| `AST_roll_5` | Numerical (Float) | Player's average Assists in the previous 5 games. | Secondary Metric |
| `STL_roll_5` | Numerical (Float) | Player's average Steals in the previous 5 games. | Secondary Metric |
| `BLK_roll_5` | Numerical (Float) | Player's average Blocks in the previous 5 games. | Secondary Metric |
| `TOV_roll_5` | Numerical (Float) | Player's average Turnovers in the previous 5 games. | Secondary Metric |
| **Context and Team Features** | | | |
| `HOME` | Categorical (Binary) | Indicator for playing at home vs. away. | Game Context |
| `Days_Rest` | Numerical (Integer) | Days since the player's last game (captures fatigue/schedule). | Game Context |
| `TEAM_PTS_roll_5` | Numerical (Float) | Player's team’s average total points scored in the previous 5 games. | Team Trend |
| `OPP_DEF_STR_roll_5` | Numerical (Float) | Opponent team's average points allowed in their last 5 games (Defensive Strength Proxy). | Opponent Context |
| `OPP_PACE_roll_5` | Numerical (Float) | Opponent team's average tempo/pace in their last 5 games (Pace Proxy). | Opponent Context |
| `OPPONENT_ID` | Categorical (Nominal) | The unique ID of the opposing team (used for categorical encoding). | Opponent Context |

---

## 4. Exploratory Data Analysis (EDA)

* **Development Environment:** Data exploration was performed using a **Jupyter Notebook** in the Google Colab environment.
* **Data Cleansing and Pre-processing:**
    * **Missing Values:** There were no missing values in the raw game log statistics. Missing values resulting from creating rolling averages (for the first four games of a player's season) were removed by dropping rows where `PTS_roll_5` was NaN. This ensured the model trained only on complete and reliable data.
    * **Tidied Dataset:** The final feature table used 17 engineered variables derived from the game logs, capturing trends and context, with each row representing one game observation.

### Summary Statistics and Visualizations

* **Key Metrics:**
    * Average points per game across the entire dataset: **11.34**.
    * Average rebounds per game: **4.28**.
    * Unique players in the dataset: **448**.
* **Target Distribution:** The distribution of the target variable (`PTS`) is strongly **right-skewed** [EDA findings in notebook after code cell 9]. This confirms that low-scoring games (0-15 points) are the most frequent outcome, while very high-scoring games (40+) are rare outliers.
* **Correlation Analysis (Key Findings from Heatmap):**
    * **Field Goals Made (FGM):** Highly correlated with Points ($\rho = 0.97$).
    * **Field Goals Attempted (FGA):** Strongly correlated with Points ($\rho = 0.89$), proving that shot volume is a powerful predictor of scoring output.
    * **Minutes Played (MIN):** Significant correlation of $\rho = 0.74$, confirming that opportunity (time on court) directly drives scoring.

---

## 5. Model Training

* **Predictive Models:** Six models were trained and evaluated:
    * Baseline (5-game historical average)
    * Linear Regression
    * Ridge Regression
    * Random Forest Regressor (Tuned)
    * XGBoost Regressor
    * Ensemble (Voting Regressor of RF + XGB + Ridge)
* **Training Strategy:**
    * **Data Split:** A chronological split was used to prevent data leakage and accurately simulate real-world forecasting.
    * **Cutoff Date:** **March 1, 2024**.
    * **Train/Test Split:** 17,211 rows (Train, pre-March 1) vs. 6,558 rows (Test, post-March 1).
    * **Python Packages:** `scikit-learn` and `xgboost` were primarily used for model training.
* **Performance Measurement:** The models were evaluated using Mean Absolute Error (MAE) and the Coefficient of Determination (R²).

---

## 6. Application of the Trained Models

* **Web Application Tool:** The project included the development of an "NBA Dashboard" web application built using **Streamlit**.
* **Functionality:** The application is designed to be interactive, allowing users to input current game data or player data and receive real-time point predictions from the chosen model.

---

## 7. Conclusion

### Summary of Work

This project successfully implemented an end-to-end data science pipeline, starting from raw NBA game logs and culminating in a set of predictive regression models and a web application. The focus on meticulous feature engineering proved highly effective in accurately modeling player performance.

### Results of Machine Learning

The primary objective of beating the baseline model was achieved. The most accurate models were the simpler linear algorithms, indicating that the relationship between the engineered features and the target is fundamentally linear.

| Model | MAE (Average Error) | R² (Variance Explained) |
| :--- | :--- | :--- |
| **Ridge Regression** | **4.78 points** | **0.509** |
| Linear Regression | 4.78 points | 0.509 |
| Ensemble (RF + XGB + Ridge) | 4.80 points | 0.508 |
| Baseline (5-game avg) | 4.91 points | 0.475 |

### Limitations

* **Prediction of Outliers:** The model's fundamental limitation is its inability to predict massive, out-of-character breakout games (e.g., a 50-point game by a low-usage player). Since the model relies on historical trends (the `PTS_roll_5` feature dominates), it can only predict the *most probable* outcome, not 1-in-a-million statistical outliers.
* **Data Granularity:** The current feature set does not include opponent individual matchups or injury status beyond `Days_Rest`, which are critical for predicting minor performance fluctuations.

### Lessons Learned

* **Feature Engineering is Paramount:** The key lesson learned was that superior feature engineering (developing features on recent trends and context) significantly outperformed complex modeling techniques (Random Forest, XGBoost). The simpler Ridge and Linear Regression models delivered the best performance, proving the engineered features accurately captured the underlying predictive signals in a linear way.

### Future Research Direction

* **Non-Linear Residual Modeling:** To address the outlier limitation, future work could focus on using a deep learning model to predict the residual error (difference) from the linear model, capturing non-linear patterns that the linear model misses.
* **Time-Series Techniques:** Explore time-series architectures (like Recurrent Neural Networks) to better model the sequence and temporal dependencies in player performance trends.
* **External Data Integration:** Incorporate external datasets such as granular player-to-player defensive match-up statistics and official injury reports to improve predictive power.

---

## 8. References

* `nba-api` Python package (Used for data collection)
* `pandas` (Used for data manipulation)
* `scikit-learn` (Used for modeling: Linear Regression, Ridge, Random Forest)
* `plotly.express` (Used for visualizations in EDA)
* `streamlit` (Used for web application development)

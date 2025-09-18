# 1. Title and Author  

**Project Title**: *Predicting NBA Player Performance: Forecasting Points Per Game with Machine Learning*  

Prepared for **UMBC Data Science Master’s Program – Capstone Project**  
Instructor: **Dr. Chaojie (Jay) Wang**  

**Author**: Kushal Senapathi 


---

# 2. Background  

Basketball analytics has rapidly evolved into a central part of decision-making in professional sports. Inspired by the “Moneyball” era, NBA teams, analysts, and fans now rely on data-driven insights to evaluate performance, optimize strategies, and gain competitive advantage.  

This project focuses on predicting **NBA player points per game (PTS)** using machine learning. By leveraging past performance metrics and contextual game factors, the model aims to forecast how many points a player will score in their next game.  

**Why it matters:**  
- Coaches can use predictions for scouting, player rotations, and game planning.  
- Fantasy sports and betting platforms can improve player projections, a multi-billion-dollar market.  
- Fans and analysts gain deeper insights into player performance trends and consistency.  

**Research Questions:**  
1. Can past performance metrics (minutes, shooting %, rebounds, assists, etc.) predict how many points a player will score in the next game?  
2. Which features (home/away status, opponent, shooting efficiency, recent rolling averages) are the strongest predictors of scoring?  
3. Which machine learning models provide the best predictive accuracy?  
4. Hypothesis: Rolling averages of player stats (last 3–5 games) will improve prediction performance compared to single-game features.  

---

# 3. Data  

### **Data Sources**  
- Official NBA Stats API ([stats.nba.com](https://www.nba.com/stats/)), collected as player game logs and stored in CSV format.  

### **Data Size**  
- ~20 MB (CSV file).  

### **Data Shape**  
- **Rows:** 23,770 (each row represents one player’s performance in a single game).  
- **Columns:** ~30 (player identifiers, game info, performance stats).  

### **Time Period**  
- October 2023 – April 2024 (2023–24 NBA season).  

### **Unit of Analysis**  
- Each row corresponds to a **player’s performance in a single NBA game**.  

### **Data Dictionary (Partial)**  

| Column Name   | Data Type   | Definition                                      | Example Values |
|---------------|------------|--------------------------------------------------|----------------|
| `PLAYER_NAME` | String     | Player full name                                | LeBron James   |
| `PLAYER_ID`   | Integer    | Unique player identifier                         | 2544           |
| `GAME_DATE`   | Date       | Date of the game                                | 2023-12-25     |
| `MATCHUP`     | String     | Team vs Opponent (Home/Away)                    | LAL vs BOS     |
| `HOME`        | String     | Derived: Home or Away game                      | Home / Away    |
| `OPPONENT`    | String     | Opponent team abbreviation                      | BOS, MIA       |
| `PTS`         | Integer    | Points scored in the game (Target variable)     | 28             |
| `REB`         | Integer    | Total rebounds in the game                      | 10             |
| `AST`         | Integer    | Total assists in the game                       | 8              |
| `MIN`         | Integer    | Minutes played                                  | 36             |
| `FG_PCT`      | Float      | Field goal percentage                           | 0.52           |
| `FG3_PCT`     | Float      | Three-point field goal percentage               | 0.40           |
| `FT_PCT`      | Float      | Free throw percentage                           | 0.85           |
| `TOV`         | Integer    | Turnovers                                       | 3              |
| `PLUS_MINUS`  | Integer    | Point differential when the player was on court | +5             |

### **Target Variable**  
- `PTS` (Points scored per game).  

### **Features/Predictors**  
- **Game-related**: `HOME`, `OPPONENT`, `GAME_DATE` (transformed into recency features).  
- **Performance stats**: `MIN`, `FG_PCT`, `FG3_PCT`, `FT_PCT`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `PLUS_MINUS`.  
- **Engineered features**: Rolling averages of points, assists, and rebounds from last 3–5 games.  

---

✅ This version is polished, professional, and **proposal-ready**.  

---

Do you want me to also draft the **first few cells of your Jupyter Notebook** (loading dataset + initial EDA) so you can place it in `/notebooks/exploration.ipynb`? That way you’ll already have a baseline to start with.

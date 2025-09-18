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

### **Data Dictionary **  

| Column Name       | Data Type  | Description                                                                 | Potential Values / Notes                         |
|------------------|------------|-----------------------------------------------------------------------------|-------------------------------------------------|
| SEASON_ID        | int64      | Unique identifier for the NBA season                                         | Example: 2023                                   |
| Player_ID        | int64      | Unique identifier for each player                                           | -                                               |
| Game_ID          | int64      | Unique identifier for each game                                             | -                                               |
| GAME_DATE        | datetime   | Date when the game was played                                               | -                                               |
| MATCHUP          | object     | Matchup info showing which teams played, e.g., 'TEAM1 vs TEAM2'            | Can parse to get Home/Away and Opponent        |
| WL               | object     | Game outcome for the player's team                                         | W = Win, L = Loss                               |
| MIN              | int64      | Minutes played by the player                                               | -                                               |
| FGM              | int64      | Field goals made                                                           | -                                               |
| FGA              | int64      | Field goals attempted                                                      | -                                               |
| FG_PCT           | float64    | Field goal percentage                                                      | FGM / FGA                                       |
| FG3M             | int64      | Three-point shots made                                                     | -                                               |
| FG3A             | int64      | Three-point shots attempted                                                | -                                               |
| FG3_PCT          | float64    | Three-point field goal percentage                                          | FG3M / FG3A                                     |
| FTM              | int64      | Free throws made                                                           | -                                               |
| FTA              | int64      | Free throws attempted                                                      | -                                               |
| FT_PCT           | float64    | Free throw percentage                                                      | FTM / FTA                                       |
| OREB             | int64      | Offensive rebounds                                                         | -                                               |
| DREB             | int64      | Defensive rebounds                                                         | -                                               |
| REB              | int64      | Total rebounds                                                             | OREB + DREB                                     |
| AST              | int64      | Assists                                                                    | -                                               |
| STL              | int64      | Steals                                                                     | -                                               |
| BLK              | int64      | Blocks                                                                     | -                                               |
| TOV              | int64      | Turnovers                                                                  | -                                               |
| PF               | int64      | Personal fouls committed                                                   | -                                               |
| PTS              | int64      | Points scored in the game                                                  | **Target variable for regression**             |
| PLUS_MINUS       | int64      | Plus-minus score, point differential when the player was on the court     | -                                               |
| VIDEO_AVAILABLE  | int64      | Indicates if video footage is available                                   | 0 = No, 1 = Yes                                |
| PLAYER_NAME      | object     | Full name of the player                                                    | -                                               |

### **Target Variable**  
- `PTS` (Points scored per game).  

### **Features/Predictors**  
- **Game-related**: `HOME`, `OPPONENT`, `GAME_DATE` (transformed into recency features).  
- **Performance stats**: `MIN`, `FG_PCT`, `FG3_PCT`, `FT_PCT`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `PLUS_MINUS`.  
- **Engineered features**: Rolling averages of points, assists, and rebounds from last 3–5 games.  

---

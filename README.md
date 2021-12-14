# Predicting the Outcome of NBA games using Machine Learning

## Instructions to run models
### Setup
```
$ cd home_team_pred
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

### To run all models
```
$ python3 predictor.py {Feature Elimination Option}
```
The options for feature elimination are as follows:
- "STANDARD": no feature elimination
- "RFE": recursive feature elimination with cross validation
- "P_VAL": statistical significance based feature elimination
- "PCA": principal component anlaysis (10 features)
- "TOP_5": top 5 features based on ANOVA F-value between label/feature for classification tasks.

### Output format
- Model Name: Model Name
- Accuracy: % of test set predicted correctly
- Confusion Matrix: counts of true positives, true negatives, false positives, and false negatives
- Precision: True Postives / True Postives + False Positives
- Recall: True Postives / True Positives + False Negatives

## Code Breakdown

### scrape.py
```
$ python3 scrape.py
```
**Description:** Scrape raw game data (date, home team, home points, visiting team, visiting points) from `https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html` for all games from the 2000-01 season to the 2018-19 season and output to `data.csv` in the format of `date, visitor_team, visitor_pts, home_team, home_pts`.

### scrape_all_stars.py
```
$ python3 scrape_all_star.py
```
**Description:** Scrape raw all-star data from `https://basketball.realgm.com/nba/allstar/game/rosters/{year}` for each year between 2000 and 2018 (we care about the all stars from the season prior to the one in consideration) and output to `all_stars.csv` in the format of `Season, Player, Team`.

### features.py
```
$ python3 features.py
```
**Description:** Featurize the raw data
The features we selected for this project are:
- `home_pts_total`: average points scored by home team in last 10 games
- `visitor_pts_total`: average points scored by visiting team in last 10 games
- `home_pts_total_as_home`: average points scored by home team in last 10 home games
- `visitor_pts_total_as_away`: average points scored by away team in last 10 away games
- `home_pts_diff`: average points differential by home team in last 10 games
- `visitor_pts_diff`: average points differential by away team in last 10 games
- `home_pts_diff_as_home`: average points differential by home team in last 10 home games
- `visitor_pts_diff_as_away`: average points differential by away team in last 10 away games
- `home_wl_pct`: win percentage by home team in last 10 games
- `visitor_wl_pct`: win percentage by away team in last 10 games
- `home_wl_pct_as_home`: win percentage by home team in last 10 games home games
- `visitor_wl_pct_as_away`: win percentage by away team in last 10 away games
- `home_play_yesterday`: 1 if home team played back-to-back, 0 if not 
- `visitor_play_yesterday`: 1 if away team played back-to-back, 0 if not 
- `home_3_in_4`: 1 if home team played 3 games in 4 days
- `visitor_3_in_4`: 1 if away team played 3 games in 4 days
- `home_2_in_3`: 1 if home team played 2 games in 3 days
- `visitor_2_in_3`: 1 if away team played 2 games in 3 days
- `home_num_stars`: number of all stars on home team roster
- `visitor_num_stars`: number of all stars on visiting team roster
- `season_WL_home`: season win percentage of home team
- `season_WL_visitor`: season win percentage of away team
- `home_WL_against_visitor`: season win percentage of home team against away team
So this python script transforms the raw data into these feature vectors and uses the game result as the label for each feature vector. At a high-level, this program works by maintaining a mapping from each team to the necessary data for the last 10 games, last 10 games at home, and last 10 games as away, updating those mappings on initial data parsing, and computing the necessary averages/percentages/date manipulations. The output of this program is `features.csv` with the feature vectors along with their associated label.

The functions within this script can be split into a few categories:
1. init: initialize various variables in dictionaries if keys are already present
1. average: compute 10-game averages of various variables
1. percenrage: compute 10-game win percentage of various variables
1. date computations: computes whether team has played back-to-back, 2 in 3, etc.
1. updates: updates the 10 game moving data stream for various variables

These functions are used to compute the various features defined above in this project.

### predictor.py
```
$ python3 predictor.py
```
**Description:** Train and test models.
This script accomplishes the following tasks:
1. Splitting `features.csv` into a test and train split of 80-20
1. Computing the accuracies and other evaluation metrics of each model
1. Computing the accuracies and other evaluation metrics for the weighted multi model
There are in-line comments that explain some of the smaller implementation details

### data_analysis.py
```
$ python3 data_analysis.py
```
**Description:** Computes the number of each class. Used to see if there are any biases in the data classifications


## Side Note about `player_stats_attempt/`
Initially, we thought we were going to try to approach the problem using both team focused models(`home_team_pred`) and a player focused models (`player_stats_attempt`). Due to time constraints, we shifted our focus solely to team focused models. That being said, to run the models using player focused data:
```
$ cd player_stats_attempt
$ python3 feature_vectors.py
```
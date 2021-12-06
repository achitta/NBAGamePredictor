import csv
from sklearn.model_selection import train_test_split
import numpy as np
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

header = ['season', 'gmDate', 'teamAbbr', 'gameId', 'teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA', 'teamFGM', 'teamFG%', 'team2PA', 'team2PM', 'team2P%', 'team3PA', 'team3PM', 'team3P%', 'teamFTA', 'teamFTM', 'teamFT%', 'teamORB', 'teamDRB', 'teamTRB', 'teamTREB%', 'teamASST%', 'teamTS%', 'teamEFG%', 'teamOREB%', 'teamDREB%', 'teamTO%', 'teamSTL%', 'teamBLK%', 'teamOrtg', 'teamDrtg', 'teamEDiff', 'teamAST/TO', 'teamSTL/TO']
team1_cols = ['teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA', 'teamFGM', 'teamFG%', 'team2PA', 'team2PM', 'team2P%', 'team3PA', 'team3PM', 'team3P%', 'teamFTA', 'teamFTM', 'teamFT%', 'teamORB', 'teamDRB', 'teamTRB', 'teamTREB%', 'teamASST%', 'teamTS%', 'teamEFG%', 'teamOREB%', 'teamDREB%', 'teamTO%', 'teamSTL%', 'teamBLK%', 'teamOrtg', 'teamDrtg', 'teamEDiff', 'teamAST/TO', 'teamSTL/TO']
team2_cols = ['opptPTS', 'opptAST', 'opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFGA', 'opptFGM', 'opptFG%', 'oppt2PA', 'oppt2PM', 'oppt2P%', 'oppt3PA', 'oppt3PM', 'oppt3P%', 'opptFTA', 'opptFTM', 'opptFT%', 'opptORB', 'opptDRB', 'opptTRB', 'opptTREB%', 'opptASST%', 'opptTS%', 'opptEFG%', 'opptOREB%', 'opptDREB%', 'opptTO%', 'opptSTL%', 'opptBLK%', 'opptOrtg', 'opptDrtg', 'opptEDiff', 'opptAST/TO', 'opptSTL/TO']
season_averages_by_game = {}
final_season_averages = {}

# Read in season averages 
with open('cleaned_data/season_averages_by_game.csv') as csvfile:
        firstLine = False
        for line in csvfile:
            if not firstLine:
                firstLine = True
                continue
            all_values = line.strip().split(',')
            season = int(all_values[0])
            gmDate = all_values[1]
            teamAbbr = all_values[2]
            gameId = int(all_values[3])
            teamAverageStats = all_values[4:]
            teamAverageStats = [float(x) for x in teamAverageStats]
            key = (season, teamAbbr,)
            if key not in season_averages_by_game:
                season_averages_by_game[key] = {}
            season_averages_by_game[key][gameId] = teamAverageStats
            final_season_averages[key] = teamAverageStats

X = []
Y = []
currentGameIds = {}
with open('cleaned_data/box_scores.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            if count % 2 == 1:
                pass
            else:
                season = int(row["season"])
                team1Abbr = row["teamAbbr"]
                key1 = (season, team1Abbr,)

                team2Abbr = row["opptAbbr"]
                key2 = (season, team2Abbr,)
                
                if key1 not in currentGameIds:
                    currentGameIds[key1] = 0
                
                if key2 not in currentGameIds:
                    currentGameIds[key2] = 0
                
                currentGameIds[key1] += 1
                currentGameIds[key2] += 1

                gameId1 = currentGameIds[key1]
                gameId2 = currentGameIds[key2]

                # TODO: FIX THIS TO BE BETTER METRIC
                if season == 2013:
                    team1Stats = final_season_averages[(season, team1Abbr,)]
                else:
                    team1Stats = final_season_averages[(season - 1, team1Abbr,)]
                    
                if gameId1 > 1:
                    team1Stats = season_averages_by_game[key1][gameId1 - 1]
                
                # TODO: FIX THIS TO BE BETTER METRIC
                if season == 2013:
                    team2Stats = final_season_averages[(season, team2Abbr,)]
                else:
                    team2Stats = final_season_averages[(season - 1, team2Abbr,)]
                if gameId2 > 1:
                    team2Stats = season_averages_by_game[key2][gameId2 - 1]

                diff_vector1 = []
                diff_vector2 = []
                for op1, op2 in zip(team1Stats, team2Stats):
                    diff_vector1.append(round(op1 - op2, 4))
                    diff_vector2.append(round(op2 - op1, 4))
                
                team1Location = 1 if row["teamLoc"] == "Home" else 0
                team2Location = 1 if row["opptLoc"] == "Home" else 0

                X.append([team1Location] + diff_vector1)
                # X.append([team2Location] + diff_vector2)
                
                result1 = float(row["teamPTS"]) - float(row["opptPTS"])
                # result2 = float(row["opptPTS"]) - float(row["teamPTS"])
                Y.append(result1)
                # Y.append(result2)
            count += 1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
X_train_transpose = np.transpose(X_train)
train_means = []
train_std = []
for i, row in enumerate(X_train_transpose):
    mean = sum(row) / len(row)
    std_dev = statistics.pstdev(row)
    train_means.append(mean)
    train_std.append(std_dev)
    for j, val in enumerate(row):
        z_score = (val - mean) / std_dev
        X_train_transpose[i][j] = z_score

X_train_normalized = np.transpose(X_train_transpose)

X_test_transpose = np.transpose(X_test)
for i, row in enumerate(X_test_transpose):
    mean = train_means[i]
    mean = train_std[i]
    for j, val in enumerate(row):
        z_score = (val - mean) / std_dev
        X_test_transpose[i][j] = z_score

X_test_normalized = np.transpose(X_test_transpose)

neigh = KNeighborsClassifier(n_neighbors=3, weights="distance")
neigh.fit(X_train_normalized, Y_train)
predictions = neigh.predict(X_test_normalized)
# clf = LogisticRegression(random_state=0).fit(X_train_normalized, Y_train)
# predictions = clf.predict(X_test_normalized)


num_correct = 0
num_total = 0
for pred, answer in zip(predictions, Y_test):
    if (pred <= 0 and answer <= 0) or (pred > 0 and answer > 0):
        num_correct += 1
    num_total += 1

print(num_correct / num_total)
print(num_total)

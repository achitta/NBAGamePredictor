import csv
import itertools
from datetime import date

def dateToSeason(date_string):
    date_obj = date.fromisoformat(date_string)
    month = date_obj.month
    year = date_obj.year
    if month > 5:
        year = year + 1
    return year

team1_cols = ['teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA', 'teamFGM', 'teamFG%', 'team2PA', 'team2PM', 'team2P%', 'team3PA', 'team3PM', 'team3P%', 'teamFTA', 'teamFTM', 'teamFT%', 'teamORB', 'teamDRB', 'teamTRB', 'teamTREB%', 'teamASST%', 'teamTS%', 'teamEFG%', 'teamOREB%', 'teamDREB%', 'teamTO%', 'teamSTL%', 'teamBLK%', 'teamOrtg', 'teamDrtg', 'teamEDiff', 'teamAST/TO', 'teamSTL/TO']
header = ['season', 'gmDate', 'teamAbbr', 'gameId', 'teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA', 'teamFGM', 'teamFG%', 'team2PA', 'team2PM', 'team2P%', 'team3PA', 'team3PM', 'team3P%', 'teamFTA', 'teamFTM', 'teamFT%', 'teamORB', 'teamDRB', 'teamTRB', 'teamTREB%', 'teamASST%', 'teamTS%', 'teamEFG%', 'teamOREB%', 'teamDREB%', 'teamTO%', 'teamSTL%', 'teamBLK%', 'teamOrtg', 'teamDrtg', 'teamEDiff', 'teamAST/TO', 'teamSTL/TO']

grouped_stats = {}
grouped_dates = {}
with open("cleaned_data/season_averages_by_game.csv", 'w') as f:
    f.write(f"{','.join(header)}\n")
    with open("cleaned_data/box_scores.csv") as csvfile:
    # with open("test.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            key = (row['season'], row['teamAbbr'])
            if key not in grouped_stats:
                grouped_stats[key] = []
                grouped_dates[key] = []
            value = []
            for col in team1_cols:
                value.append(float(row[col]))
            grouped_stats[key].append(value)
            grouped_dates[key].append(row["gmDate"])
    
    for key, all_games in grouped_stats.items():
        count = 1
        cumulative_stats = [0] * len(team1_cols)
        for game_idx, game in enumerate(all_games):
            for idx, col in enumerate(team1_cols):
                cumulative_stats[idx] += game[idx]
            averages = [round(x / count, 4) for x in cumulative_stats]
            stats_string = ",".join([str(x) for x in averages])
            f.write(f"{key[0]},{grouped_dates[key][game_idx]},{key[1]},{game_idx+1},{stats_string}\n")
            count += 1

            

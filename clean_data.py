import csv
from datetime import date
"""
Extract the columns that we will be using for analysis and disregard all other information
and add season column
"""

def dateToSeason(date_string):
    date_obj = date.fromisoformat(date_string)
    month = date_obj.month
    year = date_obj.year
    if month > 5:
        year = year + 1
    return year

columns = ['gmDate', 'teamAbbr', 'teamLoc', 'teamRslt', 'teamPTS', 'teamAST', 'teamTO', 'teamSTL', 'teamBLK', 'teamPF', 'teamFGA', 'teamFGM', 'teamFG%', 'team2PA', 'team2PM', 'team2P%', 'team3PA', 'team3PM', 'team3P%', 'teamFTA', 'teamFTM', 'teamFT%', 'teamORB', 'teamDRB', 'teamTRB', 'teamTREB%', 'teamASST%', 'teamTS%', 'teamEFG%', 'teamOREB%', 'teamDREB%', 'teamTO%', 'teamSTL%', 'teamBLK%', 'teamOrtg', 'teamDrtg', 'teamEDiff', 'teamAST/TO', 'teamSTL/TO', 'opptAbbr', 'opptLoc', 'opptRslt', 'opptPTS', 'opptAST', 'opptTO', 'opptSTL', 'opptBLK', 'opptPF', 'opptFGA', 'opptFGM', 'opptFG%', 'oppt2PA', 'oppt2PM', 'oppt2P%', 'oppt3PA', 'oppt3PM', 'oppt3P%', 'opptFTA', 'opptFTM', 'opptFT%', 'opptORB', 'opptDRB', 'opptTRB', 'opptTREB%', 'opptASST%', 'opptTS%', 'opptEFG%', 'opptOREB%', 'opptDREB%', 'opptTO%', 'opptSTL%', 'opptBLK%', 'opptOrtg', 'opptDrtg', 'opptEDiff', 'opptAST/TO', 'opptSTL/TO', 'poss', 'pace']
with open('cleaned_data/box_scores.csv', 'w') as f:
    header = ['season'] + columns
    f.write(f"{','.join(header)}\n")
    with open('data/2012-18_teamBoxScore.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            arr = []
            date_str = row['gmDate']
            season_year = dateToSeason(date_str)
            arr.append(season_year)
            for column in columns:
                arr.append(row[column])
            arr = [str(x) for x in arr]
            f.write(f"{','.join(arr)}\n")
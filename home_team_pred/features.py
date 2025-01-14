import csv
import datetime

# Key: team name
# Value: {
#   "pts_total": [last 10 game pt totals]
#   "pts_total_home": [last 10 home games pt totals]
#   "pts_total_away": [last 10 away games pt totals]
#   "pts_diff": [last 10 games pt differentials]
#   "pts_diff_home": [last 10 home games]
#   "pts_diff_away": [last 10 home away]
#   "game_dates": [dates of last 10 games]
# }
running_totals = {}

# Key: season, team
# Value: {
#   "wins": # wins,
#   "losses": # losses,
# }
season_records = {}

# Key: season, team
# Value: {
#   LAL: {
#      "wins": # wins GSW over LAL
#       "losses": # losses GSW over LAL 
#   }    
# }
season_records_against_teams = {}

# Key: season, team
# Value: number of all-stars on that team
all_star_info = {}

def read_all_star_data():
    with open("all_stars.csv", newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            season, _, team = row
            key = (season, team, )
            if key not in all_star_info:
                all_star_info[key] = 0
            all_star_info[key] += 1


def init_running_totals(team_name):
    running_totals[team_name] = {
        "pts_total": [],
        "pts_total_home": [],
        "pts_total_away": [],
        "pts_diff": [],
        "pts_diff_home": [],
        "pts_diff_away": [],
        "game_dates": []
    }

def init_season_records(season, team):
    season_records[(season, team,)] = {
        "wins": 0,
        "losses": 0
    }

def init_season_records_against_teams(season, team):
    season_records_against_teams[(season, team,)] = {}

def init_season_records_against_teams_for_matchup(season, team, oppTeam):
    season_records_against_teams[(season, team,)][oppTeam] = {
        "wins": 0,
        "losses": 0,
    }


def get_average(team_name, key):
    vals = running_totals[team_name][key]
    if len(vals) == 0:
        return 0
    return sum(vals) / len(vals)

def get_win_percentage(team_name, key):
    vals = running_totals[team_name][key]
    if len(vals) == 0:
        return 0
    numWins = 0
    for pts in vals:
        if pts > 0:
            numWins += 1
    return numWins / len(vals)

def get_average_pt_diff(team_name):
    return get_average(team_name, "pts_diff")

def get_average_pt_diff_as_home(team_name):
    return get_average(team_name, "pts_diff_home")

def get_average_pt_diff_as_away(team_name):
    return get_average(team_name, "pts_diff_away")

def get_average_win_percentage(team_name):
    return get_win_percentage(team_name, "pts_diff")

def get_average_win_percentage_as_home(team_name):
    return get_win_percentage(team_name, "pts_diff_home")

def get_average_win_percentage_as_away(team_name):
    return get_win_percentage(team_name, "pts_diff_away")

def get_average_pts_per_game(team_name):
    return get_average(team_name, "pts_total")

def get_average_pts_per_game_as_home(team_name):
    return get_average(team_name, "pts_total_home")

def get_average_pts_per_game_as_away(team_name):
    return get_average(team_name, "pts_total_away")

def datetime_to_string(date):
    return date.strftime("%Y-%m-%d")

def date_string_x_days_ago(curr_date, days_back):
    curr = datetime.datetime.fromisoformat(curr_date)
    d = datetime.timedelta(days = days_back)
    return datetime_to_string(curr - d)

def played_2_in_2(team_name, curr_date):
    yesterday = date_string_x_days_ago(curr_date, 1)
    game_dates = running_totals[team_name]["game_dates"]
    return 1 if yesterday in game_dates else 0

def played_2_in_3(team_name, curr_date):
    yesterday = date_string_x_days_ago(curr_date, 1)
    day_before_yesterday = date_string_x_days_ago(curr_date, 2)
    game_dates = running_totals[team_name]["game_dates"]
    if yesterday in game_dates or day_before_yesterday in game_dates:
        return 1
    return 0

def played_3_in_4(team_name, curr_date):
    yesterday = date_string_x_days_ago(curr_date, 1)
    day_before_yesterday = date_string_x_days_ago(curr_date, 2)
    two_days_before_yesterday = date_string_x_days_ago(curr_date, 3)
    game_dates = running_totals[team_name]["game_dates"]
    count = 0
    if yesterday in game_dates:
        count += 1
    if day_before_yesterday in game_dates:
        count += 1
    if two_days_before_yesterday in game_dates:
        count += 1
    
    if count >= 2:
        return 1
    return 0

def get_win_pct(season, team,):
    record = season_records[(season, team,)]
    wins = record['wins']
    losses = record['losses']
    if wins + losses == 0:
        return 0
    return wins / (wins + losses)

def get_win_pct_home_against_away(season, home_team, visitor_team):
    # if visitor_team not in season_records_against_teams[(season, home_team,)]:
    #     init_season_records_against_teams(season, home_team, visitor_team)

    # if home_team not in season_records_against_teams[(season, visitor_team,)]:
    #     init_season_records_against_teams(season, visitor_team, home_team)

    wins = season_records_against_teams[(season, home_team,)][visitor_team]['wins']
    losses = season_records_against_teams[(season, home_team,)][visitor_team]['losses']
    if wins + losses == 0:
        return 0
    return wins / (wins + losses)

def update_running_totals(game_date, home_team, home_pts, visitor_team, visitor_pts, max_size):
    # Evict elements if at size
    if len(running_totals[home_team]["game_dates"]) == max_size:
        running_totals[home_team]["game_dates"] = running_totals[home_team]["game_dates"][1:]

    if len(running_totals[home_team]["pts_total"]) == max_size:
        running_totals[home_team]["pts_total"] = running_totals[home_team]["pts_total"][1:]

    if len(running_totals[home_team]["pts_total_home"]) == max_size:
        running_totals[home_team]["pts_total_home"] = running_totals[home_team]["pts_total_home"][1:]

    if len(running_totals[home_team]["pts_diff"]) == max_size:
        running_totals[home_team]["pts_diff"] = running_totals[home_team]["pts_diff"][1:]

    if len(running_totals[home_team]["pts_diff_home"]) == max_size:
        running_totals[home_team]["pts_diff_home"] = running_totals[home_team]["pts_diff_home"][1:]

    if len(running_totals[visitor_team]["game_dates"]) == max_size:
        running_totals[visitor_team]["game_dates"] = running_totals[visitor_team]["game_dates"][1:]

    if len(running_totals[visitor_team]["pts_total"]) == max_size:
        running_totals[visitor_team]["pts_total"] = running_totals[visitor_team]["pts_total"][1:]

    if len(running_totals[visitor_team]["pts_total_away"]) == max_size:
        running_totals[visitor_team]["pts_total_away"] = running_totals[visitor_team]["pts_total_away"][1:]

    if len(running_totals[visitor_team]["pts_diff"]) == max_size:
        running_totals[visitor_team]["pts_diff"] = running_totals[visitor_team]["pts_diff"][1:]

    if len(running_totals[visitor_team]["pts_diff_away"]) == max_size:
        running_totals[visitor_team]["pts_diff_away"] = running_totals[visitor_team]["pts_diff_away"][1:]

    # Add new data in 
    running_totals[home_team]["game_dates"].append(game_date)
    running_totals[home_team]["pts_total"].append(home_pts)
    running_totals[home_team]["pts_total_home"].append(home_pts)
    running_totals[home_team]["pts_diff"].append(home_pts - visitor_pts)
    running_totals[home_team]["pts_diff_home"].append(home_pts - visitor_pts)

    running_totals[visitor_team]["game_dates"].append(game_date)
    running_totals[visitor_team]["pts_total"].append(visitor_pts)
    running_totals[visitor_team]["pts_total_away"].append(visitor_pts)
    running_totals[visitor_team]["pts_diff"].append(visitor_pts - home_pts)
    running_totals[visitor_team]["pts_diff_away"].append(visitor_pts - home_pts)

def update_season_records(home_team, visitor_team, season, result):
    # Home team win
    if result > 0:
        season_records[(season, home_team,)]['wins'] += 1
        season_records[(season, visitor_team,)]['losses'] += 1
    else:
        season_records[(season, visitor_team,)]['wins'] += 1
        season_records[(season, home_team,)]['losses'] += 1

def update_season_records_against_teams(home_team, visitor_team, season, result):
    # if visitor_team not in season_records_against_teams[(season, home_team,)]:
    #     init_season_records_against_teams(season, home_team, visitor_team)

    # if home_team not in season_records_against_teams[(season, visitor_team,)]:
    #     init_season_records_against_teams(season, visitor_team, home_team)

    if result > 0:
        season_records_against_teams[(season, home_team,)][visitor_team]['wins'] += 1
        season_records_against_teams[(season, visitor_team,)][home_team]['losses'] += 1
    else:
        season_records_against_teams[(season, visitor_team,)][home_team]['wins'] += 1
        season_records_against_teams[(season, home_team,)][visitor_team]['losses'] += 1

def season_from_game_date(game_date):
    gm_date = datetime.datetime.fromisoformat(game_date)
    month = gm_date.month
    year = gm_date.year
    if month <= 6:
        return str(year)
    else:
        return str(year + 1)

def get_num_all_stars(game_date, team):
    season = season_from_game_date(game_date)
    if (season, team,) not in all_star_info:
        return 0
    return all_star_info[(season, team,)]


read_all_star_data()
header = ["date","home_team","home_pts","visitor_team","visitor_pts","home_pts_total","visit_pts_total","home_pts_total_as_home","visit_pts_total_as_away","home_pts_diff","visit_pts_diff","home_pts_diff_as_home","visitor_pts_diff_as_away","home_wl_perc","visit_wl_perc","home_wl_as_home","visit_wl_as_away","home_play_yest","visit_play_yest","home_3_in_4","visit_3_in_4","home_2_in_3","visit_2_in_3","home_num_stars", "visitor_num_stars", "season_WL_home", "season_WL_visitor", "home_WL_against_visitor", "result"]
with open("features.csv", 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)
    with open("data.csv", newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            game_date = row['date']
            visitor_team = row['visitor_team']
            visitor_pts = int(row['visitor_pts'])
            home_team = row['home_team']
            home_pts = int(row['home_pts'])

            if home_team not in running_totals:
                init_running_totals(home_team)
            if visitor_team not in running_totals:
                init_running_totals(visitor_team)

            season = season_from_game_date(game_date)
            if (season, home_team,) not in season_records_against_teams:
                init_season_records_against_teams(season, home_team)
            if (season, visitor_team,) not in season_records_against_teams:
                init_season_records_against_teams(season, visitor_team)
            
            if visitor_team not in season_records_against_teams[(season, home_team,)]:
                init_season_records_against_teams_for_matchup(season, home_team, visitor_team)

            if home_team not in season_records_against_teams[(season, visitor_team,)]:
                init_season_records_against_teams_for_matchup(season, visitor_team, home_team)

            if (season, home_team,) not in season_records:
                init_season_records(season, home_team)
            if (season, visitor_team,) not in season_records:
                init_season_records(season, visitor_team)


            # Compute ppg
            home_pts_total = get_average_pts_per_game(home_team)
            visitor_pts_total = get_average_pts_per_game(visitor_team)
            home_pts_total_as_home = get_average_pts_per_game_as_home(home_team)
            visitor_pts_total_as_away = get_average_pts_per_game_as_away(visitor_team)

            # Compute ppg differential
            home_pts_diff = get_average_pt_diff(home_team)
            visitor_pts_diff = get_average_pt_diff(visitor_team)
            home_pts_diff_as_home = get_average_pt_diff_as_home(home_team)
            visitor_pts_diff_as_away = get_average_pt_diff_as_away(visitor_team)

            # Compute WL percentages
            home_wl_pct = get_average_win_percentage(home_team)
            visitor_wl_pct = get_average_win_percentage(visitor_team)
            home_wl_pct_as_home = get_average_win_percentage_as_home(home_team)
            visitor_wl_pct_as_away = get_average_win_percentage_as_away(visitor_team)

            # Compute tiredness stats
            home_played_yesterday = played_2_in_2(home_team, game_date)
            away_played_yesterday = played_2_in_2(visitor_team, game_date)
            home_3_in_4 = played_3_in_4(home_team, game_date)
            visitor_3_in_4 = played_3_in_4(visitor_team, game_date)
            home_2_in_3 = played_2_in_3(home_team, game_date)
            visitor_2_in_3 = played_2_in_3(visitor_team, game_date)

            # All star info
            home_all_stars = get_num_all_stars(game_date, home_team)
            visitor_all_stars = get_num_all_stars(game_date, visitor_team)

            win_pct_home = get_win_pct(season, home_team,)
            win_pct_away = get_win_pct(season, visitor_team)

            win_pct_home_against_away = get_win_pct_home_against_away(season, home_team, visitor_team)

            # Get result
            result = home_pts - visitor_pts

            # Write data
            data = [game_date, home_team, home_pts, visitor_team, visitor_pts, 
            home_pts_total, visitor_pts_total, home_pts_total_as_home, 
            visitor_pts_total_as_away, home_pts_diff, visitor_pts_diff, home_pts_diff_as_home, 
            visitor_pts_diff_as_away, home_wl_pct, visitor_wl_pct, home_wl_pct_as_home, 
            visitor_wl_pct_as_away, home_played_yesterday, away_played_yesterday, home_3_in_4,
            visitor_3_in_4, home_2_in_3, visitor_2_in_3, home_all_stars, visitor_all_stars, 
            win_pct_home, win_pct_away, win_pct_home_against_away, result]
            
            writer.writerow(data)

            # Update running totals
            update_running_totals(game_date, home_team, home_pts, visitor_team, visitor_pts, 10)
            update_season_records(home_team, visitor_team, season, result)
            update_season_records_against_teams(home_team, visitor_team, season, result)

            # if flag:
            #     print("AFTER UPDATE")
            #     print(season_records_against_teams[("2016", "Golden State Warriors",)]["Memphis Grizzlies"])
            #     print(season_records_against_teams[("2016", "Memphis Grizzlies",)]["Golden State Warriors"])
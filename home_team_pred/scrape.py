from bs4 import BeautifulSoup
import csv
import requests
from datetime import datetime

years = ["2001", "2002", "2003", "2004", "2005", "2006", "2007","2008", "2009", "2010", "2011",
          "2013", "2014", "2015", "2016", "2017", "2018", "2019"]

months = ["october", "november", "december", "january", "february", "march", "april", "may", "june"]

header = ["date", "visitor_team", "visitor_pts", "home_team", "home_pts"]

with open('data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for year in years:
            for month in months:
                if (year == "2005" or year == "2006") and month == "october":
                    continue
                
                request_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
                print(request_url)
                page = requests.get(request_url)
                soup = BeautifulSoup(page.content, "html.parser")
                game_dates = soup.find_all("th", {"data-stat" : "date_game"})[1:]
                visitor_names = soup.find_all("td", {"data-stat" : "visitor_team_name"})
                visitor_pts = soup.find_all("td", {"data-stat" : "visitor_pts"})
                home_names = soup.find_all("td", {"data-stat" : "home_team_name"})
                home_pts = soup.find_all("td", {"data-stat" : "home_pts"})
                
                for (a, b, c, d, e) in zip(game_dates, visitor_names, visitor_pts, home_names, home_pts):
                    game_date = a.text
                    game_date = datetime.strptime(game_date, "%a, %b %d, %Y")
                    game_date = game_date.strftime("%Y-%m-%d")
                    visitor_team = b.text
                    visitor_points_scored = int(c.text)
                    home_team = d.text
                    home_points_scored = int(e.text)
                    data = [game_date, visitor_team, visitor_points_scored, home_team, home_points_scored]
                    writer.writerow(data)

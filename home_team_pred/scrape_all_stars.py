import csv
import requests
from bs4 import BeautifulSoup


years = ["2001", "2002", "2003", "2004", "2005", "2006", "2007","2008", "2009", "2010", "2011",
          "2013", "2014", "2015", "2016", "2017", "2018", "2018", "2019"]
headers = ["Season", "Player", "Team"]
with open("all_stars.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for year in years:
        url = f"https://basketball.realgm.com/nba/allstar/game/rosters/{year}"
        print(f"Processing: {url}")
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        player_names = soup.find_all("td", {"data-th" : "Player"})
        team_names = soup.find_all("td", {"data-th" : "Team"})
        
        for (a, b) in zip(player_names, team_names):
            name = a.text.strip()
            team = b.text.strip()
            data = [year, name, team]
            writer.writerow(data)

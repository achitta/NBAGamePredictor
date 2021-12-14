def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from numpy import mean, std
import csv
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt



X = []
Y = []

header = ["date","home_team","home_pts","visitor_team","visitor_pts","home_pts_total","visit_pts_total","home_pts_total_as_home","visit_pts_total_as_away","home_pts_diff","visit_pts_diff","home_pts_diff_as_home","visitor_pts_diff_as_away","home_wl_perc","visit_wl_perc","home_wl_as_home","visit_wl_as_away","home_play_yest","visit_play_yest","home_3_in_4","visit_3_in_4","home_2_in_3","visit_2_in_3","home_num_stars", "visitor_num_stars", "season_WL_home", "season_WL_visitor", "home_WL_against_visitor", "result"]
with open("features.csv", newline='') as f:
    reader = csv.reader(f)
    next(reader)
    flag = False
    for row in reader:
        # 5,6,7,8
        x = row[5:-1]

        x = [float(a) for a in x]
        y = float(row[-1])
        
        X.append(x)
        Y.append(y)

num_wins = 0
num_total = 0
for y in Y:
    if y > 0:
        num_wins += 1
    num_total += 1
print(f"Num wins by home team: {num_wins}")
print(f"Num losses by home team: {num_total - num_wins}")
print(f"Num total: {num_total}")
print(f"Pct wins by home team: {num_wins * 100 / num_total}")
print(f"Pct losses by home team: {(num_total - num_wins) * 100 / num_total}")

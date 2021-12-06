def warn(*args, **kwargs):
    pass
import warnings
from numpy.lib.function_base import average

from sklearn import naive_bayes
warnings.warn = warn

import csv
import sys
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


X = []
Y = []
feature_select = sys.argv[1] # STANDARD, RFE, P_VALS, PCA, TOP_5
normalize = sys.argv[2] # NORM / RAW
print(f"Passed in args: {feature_select} {normalize}")

header = ["date","home_team","home_pts","visitor_team","visitor_pts","home_pts_total","visit_pts_total","home_pts_total_as_home","visit_pts_total_as_away","home_pts_diff","visit_pts_diff","home_pts_diff_as_home","visitor_pts_diff_as_away","home_wl_perc","visit_wl_perc","home_wl_as_home","visit_wl_as_away","home_play_yest","visit_play_yest","home_3_in_4","visit_3_in_4","home_2_in_3","visit_2_in_3","home_num_stars", "visitor_num_stars", "season_WL_home", "season_WL_visitor", "home_WL_against_visitor", "result"]
with open("features.csv", newline='') as f:
    reader = csv.reader(f)
    next(reader)
    flag = False
    for row in reader:
        # 5,6,7,8
        x = row[5:-1]
        
        # USING RFECV
        if feature_select == "RFE":
            rfecv_mask = [True,False,True,False,True,True,True,True,False,True,False,True,False,True,False,False,False,False,True,True,True,True,False]
            x_rfecv = []
            if not flag:
                print("RFECV: SIGNIFICANT FEATURES")
            for i, b in enumerate(rfecv_mask):
                if b:
                    if not flag:
                        print(header[i+5])
                    x_rfecv.append(x[i])
            x = x_rfecv

        # USING P VALUES 
        # 5,6,7,13,14,18,19,20,21,22
        if feature_select == "P_VAL":
            x_p_values = []
            significant_p_value_features = [4,5,6,12,13,17,18,20,21]
            if not flag:
                print("P_VALS: SIGNFICANT FEATURES")
            for p_idx in significant_p_value_features:
                if not flag:
                    print(header[p_idx + 5])
                x_p_values.append(x[p_idx])
            x = x_p_values


        flag = True
        x = [float(a) for a in x]
        y = float(row[-1])

        if feature_select == "STANDARD" and (x[0] == 0 or x[1] == 0 or x[2] == 0 or x[3] == 0):
            continue

        if feature_select == "RFE" and (x[0] == 0 or x[1] == 0):
            continue
        
        X.append(x)
        Y.append(y)

X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

Y_train_binary = []
Y_test_binary = []
for val in Y_train:
    if val > 0:
        Y_train_binary.append(1)
    else:
        Y_train_binary.append(0)

for val in Y_test:
    if val > 0:
        Y_test_binary.append(1)
    else:
        Y_test_binary.append(0)

# Normalize data
X_train_minmax = X_train_raw
X_test_minmax = X_test_raw
if normalize == "NORM":
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train_raw)
    X_test_minmax = min_max_scaler.transform(X_test_raw)

if feature_select == "PCA":
    pca = PCA(n_components=10)
    pca.fit(X_train_minmax)
    X_train_minmax = pca.transform(X_train_minmax)
    X_test_minmax = pca.transform(X_test_minmax)

if feature_select == "TOP_5":
    test = SelectKBest(k=5)
    fit = test.fit(X_train_minmax, Y_train_binary)
    X_train_minmax = fit.transform(X_train_minmax)
    X_test_minmax = fit.transform(X_test_minmax)
    arr = []

    print("TOP 5 Features selected")
    for idx, sc in enumerate(fit.scores_):
        arr.append((idx, sc, header[5:-1][idx],))
    a = sorted(arr, key= lambda x: x[1], reverse=True)
    # print(a)
    a = a[:5]
    for elt in a:
        print(elt[2])
    print()


def predictAndGetResults(model, toPrint, isRegression=False):
    print(toPrint)
    if isRegression:
        Y_train_predict = Y_train
        Y_test_predict = Y_test
    else:
        Y_train_predict = Y_train_binary
        Y_test_predict = Y_test_binary
    
    model.fit(X_train_minmax, Y_train_predict)
    predictions = model.predict(X_test_minmax)
    
    numCorrect = 0
    numTotal = 0
    totalError = 0
    for pred, answer in zip(predictions, Y_test_predict):
        if isRegression:
            if pred * answer >= 0:
                numCorrect += 1
        else:
            if pred == answer:
                numCorrect += 1
        totalError += abs(answer - pred)
        numTotal += 1 
    print(f"Accuracy: {numCorrect/numTotal}")
    if isRegression:
        print(f"Average error in pt differential: {totalError / numTotal}")
    print()

def multipleModelPrediction(models, toPrint, isRegression=False):
    print(toPrint)
    if isRegression:
        Y_train_predict = Y_train
        Y_test_predict = Y_test
    else:
        Y_train_predict = Y_train_binary
        Y_test_predict = Y_test_binary
    
    all_predictions = []
    for model in models:
        model.fit(X_train_minmax, Y_train_predict)
        predictions = model.predict(X_test_minmax)
        all_predictions.append(predictions)
    
    f1_scores = []
    for predictions in all_predictions:
        score = f1_score(Y_test_predict, predictions)
        f1_scores.append(score)
    print(f1_scores)
    score_sum = sum(f1_scores)
    voting_weights = [x / score_sum for x in f1_scores]
    
    final_predictions = []
    numCols = len(models)
    numRows = len(Y_test_predict)
    for row in range(numRows):
        weighted_prediction = 0
        for col in range(numCols):
            weighted_prediction += (voting_weights[col] * all_predictions[col][row])
        if weighted_prediction > 0.5:
            final_predictions.append(1)
        else:
            final_predictions.append(0)

    numCorrect = 0
    numTotal = 0
    totalError = 0
    for pred, answer in zip(final_predictions, Y_test_predict):
        if isRegression:
            if pred * answer >= 0:
                numCorrect += 1
        else:
            if pred == answer:
                numCorrect += 1
        totalError += abs(answer - pred)
        numTotal += 1 
    print(f"Accuracy: {numCorrect/numTotal}")
    if isRegression:
        print(f"Average error in pt differential: {totalError / numTotal}")
    score = f1_score(Y_test_predict, predictions)
    print(f"F1_Score: {score}")
    print()

print("===========================================================")
randForest = RandomForestClassifier()
predictAndGetResults(randForest, "Random Forest")

logReg = LogisticRegression()
predictAndGetResults(logReg, "Logistical Regression - Classification")

logReg = LogisticRegression()
predictAndGetResults(logReg, "Logistical Regression - Regression", isRegression=True)

linReg = LinearRegression()
predictAndGetResults(linReg, "Linear Regression - Classification")

linReg = LinearRegression()
predictAndGetResults(linReg, "Linear Regression - Regression", isRegression=True)

knn = KNeighborsClassifier(n_neighbors=3)
predictAndGetResults(knn, "KNN (3) - Classification")

knn = KNeighborsClassifier(n_neighbors=5)
predictAndGetResults(knn, "KNN (5) - Classification")

knn = KNeighborsClassifier(n_neighbors=7)
predictAndGetResults(knn, "KNN (7) - Classification")

knn = KNeighborsClassifier(n_neighbors=9)
predictAndGetResults(knn, "KNN (9) - Classification")

knn = KNeighborsClassifier(n_neighbors=11)
predictAndGetResults(knn, "KNN (11) - Classification")

knn = KNeighborsClassifier(n_neighbors=13)
predictAndGetResults(knn, "KNN (13) - Classification")

knn = KNeighborsClassifier(n_neighbors=15)
predictAndGetResults(knn, "KNN (15) - Classification")

naiveBayes = GaussianNB()
predictAndGetResults(naiveBayes, "Naive Bayes - Classification")

decisionTree = DecisionTreeClassifier()
predictAndGetResults(decisionTree, "Decision Tree - Classification")

randForest = RandomForestClassifier()
logReg = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=11)
naiveBayes = GaussianNB()
multipleModelPrediction([randForest, logReg, knn, naiveBayes], "Rand Forest + Log Reg + KNN (11) + NB")



# # Code for plotting data for the home_pts_diff and home_pts_diff_as_home against result
# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111, projection='3d')
# fourthColumn = []
# sixthColumn = []
# for idx, row in enumerate(X_train_minmax):
#     if idx % 50 == 0:
#         ax.scatter(row[0], row[1], Y_train[idx])
# plt.show()


# # Code for generating feature coefficients graph for rfe
# clf = RandomForestClassifier()
# cv_selector = RFECV(clf)
# cv_selector = cv_selector.fit(X_train_minmax, Y_train_binary)
# rfecv_mask = cv_selector.get_support() #list of booleans
# rfecv_features = [] 
# coefs = []
# for bool, feature, coeff in zip(rfecv_mask, header[5:-1], clf.coef_[0]):
#     if bool:
#         rfecv_features.append(feature)
#         coefs.append(coeff)

# print(f"Optimal number of features : {cv_selector.n_features_}")
# print(f"Best features : {rfecv_features}")
# n_features = X_train_minmax.shape[1]
# plt.figure(figsize=(8,8))
# plt.barh(range(n_features), coeff, align='center') 
# # plt.yticks(np.arange(n_features), X_train_minmax.columns.values) 
# plt.xlabel('Feature importance')
# plt.ylabel('Feature')
# plt.show()

# # Code to generate p-vals
# import statsmodels.api as sm
# logit_model=sm.Logit(Y_train_binary,X_train_minmax)
# result=logit_model.fit()
# print(result.summary())
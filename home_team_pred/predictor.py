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
try:
    feature_select = sys.argv[1] # STANDARD, RFE, P_VAL, PCA, TOP_5
except:
    feature_select = "STANDARD"

print(f"Passed in args: {feature_select}")

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

        # Data cleaning: If any of home_pts_total, visit_points_total, home_points_as_home,or visitor_points_as_away are 0
        # then we want to ignore this data point. Considering that these values would never be 0 in a real NBA setting, 
        # they might throw off the normalization factors and the rest of the prediciton. To avoid issues, we simply threw
        # these data points out (less than 50 in the overall dataset of ~24,500 data points)
        if feature_select == "STANDARD" and (x[0] == 0 or x[1] == 0 or x[2] == 0 or x[3] == 0):
            continue

        if feature_select == "RFE" and (x[0] == 0 or x[1] == 0):
            continue
        
        X.append(x)
        Y.append(y)

# 80-20 test train split
X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Create binary representations of the labels (win = 1, loss = 0)
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
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train_raw)
X_test_minmax = min_max_scaler.transform(X_test_raw)

# Feature elim using PCA
if feature_select == "PCA":
    pca = PCA(n_components=10)
    pca.fit(X_train_minmax)
    X_train_minmax = pca.transform(X_train_minmax)
    X_test_minmax = pca.transform(X_test_minmax)

# Feature elim using top 5 based on ANOVA F-test
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

# Train the model on the training data and compute accuracy and confusion matrix on test set
def predictAndGetResults(model, toPrint, isRegression=False):
    print(toPrint)
    if isRegression:
        Y_train_predict = Y_train
        Y_test_predict = Y_test
    else:
        Y_train_predict = Y_train_binary
        Y_test_predict = Y_test_binary
    
    # Train model and generate predictions
    model.fit(X_train_minmax, Y_train_predict)
    predictions = model.predict(X_test_minmax)
    
    # Compute accuracy
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
        # Compute R^2 and Adjusted R^2 for linear regression
        print(f"Average error in pt differential: {totalError / numTotal}")
        r_score = r2_score(Y_test_predict, predictions)
        print(f"Unadjusted R^2: {r_score}")
        n = len(predictions)
        k = len(X_train_minmax[0])
        adjusted_r2 = 1-((1 - r_score) * (n-1) / (n-k-1))
        print(f"Adjusted R^2: {adjusted_r2}")
    else:
        # Compute confusion matrix for all classification models
        cm = confusion_matrix(Y_test_predict, predictions)
        tn, fp, fn, tp = cm.ravel()
        print("Confusion Matrix")
        print("=============================================")
        print(f"True Positives: {tp}  | True Negatives: {tn}")
        print("---------------------------------------------")
        print(f"False Positives: {fp} | False Negatives: {fn}")
        print()
        print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn)}")
        print(f"Precision: {(tp)/(tp + fp)}")
        print(f"Recall: {(tp)/(tp + fn)}")
        print()

        # Use KFold cross validation as another validation step 
        cv = KFold(n_splits=5, random_state=1, shuffle=True)
        scores = cross_val_score(model, X, [1 if y > 0 else 0 for y in Y], scoring='accuracy', cv=cv, n_jobs=-1)
        print('K_FOLD Accuracy (STD_DEV): %.3f (%.3f)' % (mean(scores), std(scores)))
        print()
    print()

# Generate prediction accuracies for weighted multi-model
def multipleModelPrediction(models, toPrint, isRegression=False):
    print(toPrint)
    if isRegression:
        Y_train_predict = Y_train
        Y_test_predict = Y_test
    else:
        Y_train_predict = Y_train_binary
        Y_test_predict = Y_test_binary
    
    X_train_minmax_cp = X_train_minmax.copy()
    Y_train_cp = Y_train_predict.copy()
    X_train_train, X_train_validation, Y_train_train, Y_train_validation = train_test_split(X_train_minmax_cp, Y_train_cp, test_size=0.20, random_state=12)

    all_predictions = []
    for model in models:
        model.fit(X_train_train, Y_train_train)
        predictions = model.predict(X_train_validation)
        all_predictions.append(predictions)
    
    # Use f1_scores to assign voting power to each underlying model
    f1_scores = []
    for predictions in all_predictions:
        score = f1_score(Y_train_validation, predictions)
        f1_scores.append(score)
    score_sum = sum(f1_scores)
    voting_weights = [x / score_sum for x in f1_scores]
    
    all_predictions = []
    for model in models:
        model.fit(X_train_train, Y_train_train)
        predictions = model.predict(X_test_minmax)
        all_predictions.append(predictions)
    # Generate a weighted prediction
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

    # Calculate accuracy
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
    cm = confusion_matrix(Y_test_predict, predictions)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix")
    print("=============================================")
    print(f"True Positives: {tp}  | True Negatives: {tn}")
    print("---------------------------------------------")
    print(f"False Positives: {fp} | False Negatives: {fn}")
    print()
    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn)}")
    print(f"Precision: {(tp)/(tp + fp)}")
    print(f"Recall: {(tp)/(tp + fn)}")
    print()
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, [1 if y > 0 else 0 for y in Y], scoring='accuracy', cv=cv, n_jobs=-1)
    print('K_FOLD Accuracy (STD_DEV): %.3f (%.3f)' % (mean(scores), std(scores)))
    print()

print("===========================================================")
randForest = RandomForestClassifier()
predictAndGetResults(randForest, "Random Forest")

logReg = LogisticRegression()
predictAndGetResults(logReg, "Logistical Regression - Classification")

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

# # Code to generate p-vals
# import statsmodels.api as sm
# logit_model=sm.Logit(Y_train_binary,X_train_minmax)
# result=logit_model.fit()
# print(result.summary())

# # Code to generate feature importance graphs
# feature_names = [header[i+5] for i in range(X_train_minmax.shape[1])]
# forest = RandomForestClassifier(random_state=0)
# forest.fit(X_train_minmax, Y_train_binary)
# importances = forest.feature_importances_
# std_dev = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
# forest_importances = pd.Series(importances, index=feature_names)

# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std_dev, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
# plt.show()
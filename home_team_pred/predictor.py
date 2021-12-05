import csv
from sklearn.model_selection import train_test_split
import numpy as np
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

X = []
Y = []
with open("features.csv", newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        x = row[5:-1]
        x = [float(a) for a in x]
        y = float(row[-1])
        X.append(x)
        Y.append(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = LogisticRegression().fit(X_train, Y_train)
predictions = clf.predict(X_test)
# neigh = KNeighborsClassifier(n_neighbors=3, weights="distance")
# neigh.fit(X_train, Y_train)
# predictions = neigh.predict(X_test)
numCorrect = 0
numTotal = 0
for pred, answer in zip(predictions, Y_test):
    if pred * answer >= 0:
        numCorrect += 1
    numTotal += 1 
print(numCorrect/numTotal)
print(numTotal)

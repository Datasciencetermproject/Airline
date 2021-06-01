import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


y = pd.read_csv("airline/data/" + "target.csv")['satisfaction']

X=[]
X.append(pd.read_csv("airline/data/" + "scaled_data0.csv"))
X.append(pd.read_csv("airline/data/" + "scaled_data1.csv"))
X.append(pd.read_csv("airline/data/" + "scaled_data2.csv"))
X.append(pd.read_csv("airline/data/" + "scaled_data3.csv"))
X.append(pd.read_csv("airline/data/" + "scaled_data4.csv"))
X.append(pd.read_csv("airline/data/" + "scaled_data5.csv"))


print("D")
for i in range(0,6):
    train_x, test_x, train_y, test_y = train_test_split(X[i], y, test_size=0.3, random_state=1)
    param = {'max_depth':[10,20,30]}

    model = DecisionTreeClassifier()
    gscv = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1)
    gscv.fit(train_x,train_y)
    print("scaled data", i)
    print("best param: ",gscv.best_params_)
    print("best score: ",gscv.best_score_)
    print("\n\n")

print("Bagging")
for i in range(0,6):
    train_x, test_x, train_y, test_y = train_test_split(X[i], y, test_size=0.3, random_state=1)
    param = {'n_estimators':[50,100,200,300]}

    model = BaggingClassifier()
    gscv = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1)
    gscv.fit(train_x,train_y)
    print("scaled data", i)
    print("best param: ",gscv.best_params_)
    print("best score: ",gscv.best_score_)
    print("\n\n")

print("RandomForest")
for i in range(0,1):
    train_x, test_x, train_y, test_y = train_test_split(X[i], y, test_size=0.3, random_state=1)
    param = {'n_estimators':[200],
            'criterion' : ['entropy'],
            }

    model = RandomForestClassifier()
    gscv = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1)
    gscv.fit(train_x,train_y)
    print("scaled data", i)
    print("best param: ",gscv.best_params_)
    print("best score: ",gscv.best_score_)
    print("\n\n")

print("AdaBoost")
for i in range(0,6):
    train_x, test_x, train_y, test_y = train_test_split(X[i], y, test_size=0.3, random_state=1)
    param = {'n_estimators':[25,50,100,200]}

    model = AdaBoostClassifier()
    gscv = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1)
    gscv.fit(train_x,train_y)
    print("scaled data", i)
    print("best param: ",gscv.best_params_)
    print("best score: ",gscv.best_score_)
    print("\n\n")

print("GradientBoost")
for i in range(0,6):
    train_x, test_x, train_y, test_y = train_test_split(X[i], y, test_size=0.3, random_state=1)
    param = {'n_estimators':[25,50,100,200]}

    model = GradientBoostingClassifier()
    gscv = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1)
    gscv.fit(train_x,train_y)
    print("scaled data", i)
    print("best param: ",gscv.best_params_)
    print("best score: ",gscv.best_score_)
    print("\n\n")

print("XGBoost")
for i in range(0,6):
    train_x, test_x, train_y, test_y = train_test_split(X[i], y, test_size=0.3, random_state=1)
    param = {}

    model = XGBClassifier()
    gscv = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1)
    gscv.fit(train_x,train_y)
    print("scaled data", i)
    print("best param: ",gscv.best_params_)
    print("best score: ",gscv.best_score_)
    print("\n\n")

print("KNN")
for i in range(0,6):
    train_x, test_x, train_y, test_y = train_test_split(X[i], y, test_size=0.3, random_state=1)
    param = {'n_neighbors':[5,10,15,20],
            'algorithm':['ball_tree','kd_tree'],
            'weight':['uniform','distance']}

    model = KNeighborsClassifier()
    gscv = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1)
    gscv.fit(train_x,train_y)
    print("scaled data", i)
    print("best param: ",gscv.best_params_)
    print("best score: ",gscv.best_score_)
    print("\n\n")

print("Linear Regression")
for i in range(0,6):
    train_x, test_x, train_y, test_y = train_test_split(X[i], y, test_size=0.3, random_state=1)

    model = LinearRegression()
    gscv = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1)
    gscv.fit(train_x,train_y)
    print("scaled data", i)
    print("best param: ",gscv.best_params_)
    print("best score: ",gscv.best_score_)
    print("\n\n")

print("Logistic Regression")
for i in range(0,6):
    train_x, test_x, train_y, test_y = train_test_split(X[i], y, test_size=0.3, random_state=1)

    model = LogisticRegression()
    gscv = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1)
    gscv.fit(train_x,train_y)
    print("scaled data", i)
    print("best param: ",gscv.best_params_)
    print("best score: ",gscv.best_score_)
    print("\n\n")




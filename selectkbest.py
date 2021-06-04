import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

file = "airline/data/"
data0=pd.read_csv(file + "scaled_data0.csv")
data3=pd.read_csv(file +"scaled_data3.csv")

X=data0.iloc[:,0:22]
y=data0.iloc[:,-1]

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfcolumns = pd.DataFrame(X.columns)
dfscores = pd.DataFrame(fit.scores_)
#concatenate two dataframes for better visualization 
featureScores = pd.concat([dfcolumns, dfscores],axis=1)

featureScores.columns = ['Specs',"Score"] #name the dataframe columns
print(featureScores.nlargest(10,'Score')) #print 10 best features

X=data3.iloc[:,0:22]
y=data3.iloc[:,-1]

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfcolumns = pd.DataFrame(X.columns)
dfscores = pd.DataFrame(fit.scores_)
#concatenate two dataframes for better visualization 
featureScores = pd.concat([dfcolumns, dfscores],axis=1)

featureScores.columns = ['Specs',"Score"] #name the dataframe columns
print(featureScores.nlargest(10,'Score')) #print 10 best features
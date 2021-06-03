from typing import List
import numpy as np
import pandas as pd
from pandas.core import algorithms
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

class modelScore:
    """
    modelScore

    modelScore class is class to save formed data and sort formed data
    this class consists of score, scaler, encoder, model
    """
    def __init__(self, score, scaler, encoder, model) -> None:
        self.score = score
        self.scaler = scaler
        self.encoder = encoder
        self.model = model

    def __lt__(self, other):
        return self.score > other.score

    def getString(self) -> str:
        return "score: " + str(self.score) + " scaler: " + str(self.scaler) + " encoder: " + str(self.encoder) + " model: " + str(self.model)


def FindBestofBest(X,y, encode_col, scaled_col, scalers = None, encoders = None, models = None) :
    """
    Find the best combination of scaler, encoder, fitting algoritm
    print best score and best combination

    Parameters
    --------------------------------
    X: DataFrame to scaled

    y: DataFrame to encoding

    encode_col: columns to encode

    scaled_col: columns to scaled

    scalers: list of sclaer
            None: [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
            if you want to scale other ways, then put the sclaer in list

    encoders: list of encoder
        None: [OrdinalEncoder(), OneHotEncoder()]
        if you want to use only one, put a encoder in list

    models: list of encoder
        None: [AdaBoostClassifier(), BaggingClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]
        if you want to fitting other ways, then put in list
    """
    
    if encoders == None:
         encode = [OrdinalEncoder(), OneHotEncoder()]
    else: encode = encoders

    if scalers == None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else: scale = scalers

    if models == None:
        classifier = [AdaBoostClassifier(), BaggingClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]
    else: classifier = models
    
    best_score = 0
    best_of_best = {}
    scores = []
    for i in scale :
        for j in encode :
            scaler = i
            scaler = pd.DataFrame(scaler.fit_transform(X[scaled_col]))
            
            if j == OrdinalEncoder():
                enc = j
                enc = enc.fit_transform(X[encode_col])
                new_df = pd.concat([scaler, enc], axis=1)
            else:
                dum = pd.DataFrame(pd.get_dummies(X[encode_col]))                
                new_df =pd.concat([scaler, dum], axis=1)
            
    
        
            for model in classifier:
               
                X_train, X_test, y_train, y_test = train_test_split(new_df,y)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)

                item = modelScore(score,i,j,model)
                scores.append(item)

                if best_score == 0 or best_score < score:
                    best_score = score
                    best_of_best['classifier'] = model
                    best_of_best['scaler'] = i
                    best_of_best['encoder'] = j
    
    print("top 5:")
    sorted_list = sorted(scores)
    for n in range(0,5):
        print(sorted_list[n].getString())           
 
    return


def encodeNscale(X, encode_col, scaled_col, scalers = None, encoders = None) -> List :
    """
    Find the best combination of scaler, encoder, fitting algoritm
    print best score and best combination

    Parameters
    --------------------------------
    X: DataFrame to scaled and encode

    encode_col: columns to encode

    scaled_col: columns to scaled

    scaler: list of sclaer
            None: [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
            if you want to scale other ways, then put the sclaer in list

    encoder: list of encoder
        None: [OrdinalEncoder(), OneHotEncoder()]
        if you want to use only one, put a encoder in list

    Return
    -------------------------------------
    List of DataFrame that scaled and encoded

    """
    
    if encoders == None:
         encode = [OrdinalEncoder(), OneHotEncoder()]
    else: encode = encoders

    if scalers == None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else: scale = scalers

    df_list = []
    for i in scale :
        for j in encode :
            scaler = i
            scaler = pd.DataFrame(scaler.fit_transform(X[scaled_col]))
            
            if j == OrdinalEncoder():
                enc = j
                enc = enc.fit_transform(X[encode_col])
                new_df = pd.concat([scaler, enc], axis=1)
            else:
                dum = pd.DataFrame(pd.get_dummies(X[encode_col]))                
                new_df =pd.concat([scaler, dum], axis=1)
            
            df_list.append(new_df)
    
    return df_list
    



#---------------------------------------------------------------------------------------------------
df = pd.read_csv("data/airline.csv")

numeric_mask = [ 'Age', 'Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient',
       'Food and drink', 'Gate location', 'Inflight wifi service',
       'Inflight entertainment', 'Online support', 'Ease of Online booking',
       'On-board service', 'Leg room service', 'Baggage handling',
       'Checkin service', 'Cleanliness', 'Online boarding',
       'Departure Delay in Minutes', 'Arrival Delay in Minutes']

category_mask = ['Gender','Customer Type','Type of Travel','Class']


df.fillna(method='bfill',inplace=True)
df.info()
X = df.drop(columns='satisfaction')
y = df['satisfaction'].replace({'satisfied': 0, 'dissatisfied': 1})

dfs = encodeNscale(X,category_mask,numeric_mask)
for data in dfs:
    data.info()
    print(data.describe())

# FindBestofBest(X,y,category_mask,numeric_mask)

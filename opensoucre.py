import numpy as np
import pandas as pd
from pandas.core import algorithms
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV



def FindBestofBest(X,y, encode_col, scaled_col, to_scale = None, to_encode = None, to_fit = None) :
    """
    Find the best combination of scaler, encoder, fitting algoritm
    print best score and best combination

    Parameters
    --------------------------------
    ForScaledDf: DataFrame to scaled

    ForEncodingDf: DataFrame to encoding

    TargetDf: DataFrame of target

    to_scale: list of sclaer
            None: [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
            if you want to scale other ways, then put the sclaer in list

    to_encode: list of encoder
        None: [OrdinalEncoder(), OneHotEncoder()]
        if you want to encoder other ways, then put the encoder in list

    to_fit: list of encoder
        None: [AdaBoostClassifier(), BaggingClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]
        if you want to fitting other ways, then put in list
    """
    
    if to_encode == None:
         encode = [OrdinalEncoder(), OneHotEncoder()]
    else: encode = to_encode

    if to_scale == None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else: scale = to_scale

    if to_fit == None:
        classifier = [AdaBoostClassifier(), BaggingClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]
    else: classifier = to_fit
    
    best_score = 0
    best_of_best = {}
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

                if best_score == 0 or best_score < score:
                    best_score = score
                    best_of_best['classifier'] = model
                    best_of_best['scaler'] = i
                    best_of_best['encoder'] = j
                       
    
    
    print("best score", best_score)
    print("best of best", best_of_best)
    return


df = pd.read_csv("airline/data/airline.csv")

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


FindBestofBest(X,y,category_mask,numeric_mask)

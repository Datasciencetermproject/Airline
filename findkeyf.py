from typing import List
from pandas.core.frame import DataFrame
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split




def upService(data: DataFrame, up) -> DataFrame:
    """
    upService

    return a DataFrame that increase a service score

    Parameters
    ----------------------------------
    data: DataFrame
        the original data
    up: index of column
    """
    df = pd.DataFrame.copy(data, deep=True)

    df[str(up)] += 0.8
    return df

def downService(data: DataFrame, down) -> DataFrame:
    """
    downService

    return a DataFrame that decrease a service score

    Parameters
    ----------------------------------
    data: DataFrame
        the original data
    down: index of column
    """
    df = pd.DataFrame.copy(data, deep=True)

    df[str(down)] -= 0.8
    return df

def modService(data: DataFrame, uplist, downlist, upp = 0.8, downp = 0.8) -> DataFrame:
    """
    modervice

    return a DataFrame that increase or decrease services score

    Parameters
    ----------------------------------
    data: DataFrame
        the original data
    uplist: list of to increase services
    downlist: list of to decrease services
    upp: point to increase
    downp: point to decrease
    """
    df = pd.DataFrame.copy(data, deep=True)

    for n in uplist:
        df[str(n)] += upp

    for n in downlist:
        df[str(n)] -= downp
    
    return df



# ---------------------------------------------------------------------------
file = "airline/data/scaled_data"
csv = ".csv"

y = pd.read_csv("airline/data/target.csv")['satisfaction']
X = pd.read_csv( "airline/data/scaled_data2.csv")

print(y.value_counts())
print(y.value_counts()[0])
model = []
model.append(BaggingClassifier(n_estimators=200))
model.append(RandomForestClassifier(n_estimators=300,criterion='entropy'))
model.append(XGBClassifier(use_label_encoder=False))

for m in model:
    m.fit(X,y)


print(y.value_counts()[0]/len(y))
# service: 2 to 15
# to find key feature to impact satisfaction
for service in range(2,16):
    print("service:", service)

    newTest = upService(X,service)

    for mo in model:
        newY = pd.Series(mo.predict(newTest))
        print(newY.value_counts()[0]/len(newY))

# to find features not to impact satisfaction
for service in range(2,16):
    print("service:", service)

    newTest = downService(X,service)

    for mo in model:
        newY = pd.Series(mo.predict(newTest))
        print(newY.value_counts()[0]/len(newY))



# find best combination to increase satisfaction
uplist = [2,7,12,13,14]
downlist = [3,4,8,9,10,11]

newTest = modService(X,uplist,[])
for mo in model:
        newY = pd.Series(mo.predict(newTest))
        print(newY.value_counts()[0]/len(newY))
        
newTest = modService(X,[],downlist)
for mo in model:
        newY = pd.Series(mo.predict(newTest))
        print(newY.value_counts()[0]/len(newY))
        
newTest = modService(X,uplist,downlist)
for mo in model:
        newY = pd.Series(mo.predict(newTest))
        print(newY.value_counts()[0]/len(newY))

newTest = modService(X,[7,12,13,14],downlist)
for mo in model:
        newY = pd.Series(mo.predict(newTest))
        print(newY.value_counts()[0]/len(newY))
        
newTest = modService(X,[2,7,13,14],downlist)
for mo in model:
        newY = pd.Series(mo.predict(newTest))
        print(newY.value_counts()[0]/len(newY))
        
newTest = modService(X,[2,7],downlist)
for mo in model:
        newY = pd.Series(mo.predict(newTest))
        print(newY.value_counts()[0]/len(newY))

newTest = modService(X,[7,2,12],downlist)
for mo in model:
        newY = pd.Series(mo.predict(newTest))
        print(newY.value_counts()[0]/len(newY))

newTest = modService(X,[2,7,14],downlist)
for mo in model:
        newY = pd.Series(mo.predict(newTest))
        print(newY.value_counts()[0]/len(newY))

newTest = modService(X,[2,7,13],downlist)
for mo in model:
        newY = pd.Series(mo.predict(newTest))
        print(newY.value_counts()[0]/len(newY))
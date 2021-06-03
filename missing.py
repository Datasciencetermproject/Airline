from os import replace
import pandas as pd
import numpy as np
import random

from pandas.core.frame import DataFrame


def makeToDirty(percent: float, data: pd.DataFrame) -> pd.DataFrame:
    """
    makeToDirty

    add dirty data to orinal data

    Parameters
    ---------------------------
    percent: float
        rate of adding dirty data(%)
    data: DataFrame
        original data to add dirty datat

    Return
    ----------------------------
    DataFrame of added dirty data
    """
    missData = data.copy(deep=True)
    numMissing = int(len(missData) * percent/ 100)
    columnNum = missData.columns
    print("number of added dirty data: " , numMissing )

     # make nan
    for num in range(0,int(numMissing/2)):
        missing_data = missData.iloc[random.randrange(0, len(missData))]
        for i in range(0,random.randint(0,5)):
            rand = random.randrange(0,len(columnNum))
            missing_data[rand] = np.NaN
       
        missData = missData.append(missing_data)
    
    # make wrong
    for num in range(0,int(numMissing/2)):
        missing_data = missData.iloc[random.randrange(0, len(missData))]
        for i in range(0,random.randint(0,3)):
            rand = random.randrange(0,len(columnNum))
            missing_data[rand] = -1
       
        missData = missData.append(missing_data)

    return missData



data = pd.read_csv("airline/airline.csv")
data.info()
missData = makeToDirty(1,data)
missData.info()
# missData.to_csv('airline/missedAirline.csv',index=False,)


missData.replace(["-1",-1],np.nan)
missData.fillna(method="bfill",inplace=True)
missData.info()
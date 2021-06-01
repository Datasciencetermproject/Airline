from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 

numeric_mask = [ 'Age', 'Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient',
       'Food and drink', 'Gate location', 'Inflight wifi service',
       'Inflight entertainment', 'Online support', 'Ease of Online booking',
       'On-board service', 'Leg room service', 'Baggage handling',
       'Checkin service', 'Cleanliness', 'Online boarding',
       'Departure Delay in Minutes', 'Arrival Delay in Minutes']

# read csv
df = pd.read_csv('airline/data/airline.csv')

# missing value
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].mean())

# predict value
df['satisfaction'] = df['satisfaction'].replace({'satisfied': 0,'dissatisfied': 1})

# categorical value
#ordinal encoding
enc = OrdinalEncoder()
X = df.loc[:,('Gender','Customer Type','Type of Travel','Class')]
X = X.dropna(how='any')
enc.fit(X)
df.loc[:,('Gender','Customer Type','Type of Travel','Class')]=enc.transform(X)

# Using OneHotEncoder
dummies = pd.get_dummies(df, columns=['Gender','Customer Type','Type of Travel','Class'],
                                      prefix=('Gender','Customer Type','Type of Travel','Class'))


# Scaling
numericX = df[numeric_mask]
oneX = dummies.drop(columns='satisfaction').drop(columns=numeric_mask)
enX = df.drop(columns='satisfaction').drop(columns=numeric_mask)

numericX.info()
oneX.info()
enX.info()

y = df['satisfaction']

# Scaler
scaler_robust = RobustScaler()
scaler_minmax = MinMaxScaler()
scaler_stand = StandardScaler()

scaled_data = []
scaled_data.append(pd.concat([pd.DataFrame(scaler_minmax.fit_transform(numericX)),oneX],axis=1))
scaled_data.append(pd.concat([pd.DataFrame(scaler_robust.fit_transform(numericX)),oneX],axis=1))
scaled_data.append(pd.concat([pd.DataFrame(scaler_stand.fit_transform(numericX)),oneX],axis=1))
scaled_data.append(pd.concat([pd.DataFrame(scaler_minmax.fit_transform(numericX)),enX],axis=1))
scaled_data.append(pd.concat([pd.DataFrame(scaler_robust.fit_transform(numericX)),enX],axis=1))
scaled_data.append(pd.concat([pd.DataFrame(scaler_stand.fit_transform(numericX)),enX],axis=1))

# save data
i = 0
for data in scaled_data:
        data = pd.DataFrame(data)
        data.to_csv("airline/data/scaled_data" + str(i) +".csv")
        i += 1





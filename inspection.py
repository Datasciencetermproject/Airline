import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
import seaborn as sns

# read datasets
y=pd.read_csv("data/" + "target.csv")

X = []
X.append(pd.read_csv("data/" + "scaled_data0.csv"))
X.append(pd.read_csv("data/" + "scaled_data1.csv"))
X.append(pd.read_csv("data/" + "scaled_data2.csv"))
X.append(pd.read_csv("data/" + "scaled_data3.csv"))
X.append(pd.read_csv("data/" + "scaled_data4.csv"))
X.append(pd.read_csv("data/" + "scaled_data5.csv"))


# decision tree -----------------------------------------
for n in range(0,6):
    x=X[n].iloc[:,0:22]
    x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.20)
    dt_clf = DecisionTreeClassifier(max_depth=3, random_state=156)
    dt_clf.fit(x_train , y_train)
    print(dt_clf.score(x_test,y_test))
    tree.plot_tree(dt_clf)
    plt.show()

# heatmap -----------------------------------------
for n in range(0,6):
    x=X[n].iloc[:,0:22]
    data = pd.concat([x,y],axis=1)
    sns.heatmap(data.corr(),annot=True)
    plt.show()

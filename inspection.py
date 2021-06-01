import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree
import seaborn as sns

# decision tree -----------------------------------------
file = "airline/data/"
data0=pd.read_csv(file + "scaled_data2.csv")

x=data0.iloc[:,0:22]
y=pd.read_csv(file + "target.csv")


x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.20)


dt_clf = DecisionTreeClassifier(max_depth=3, random_state=156)
dt_clf.fit(x_train , y_train)
print(dt_clf.score(x_test,y_test))


tree.plot_tree(dt_clf)
plt.show()

# heatmap -----------------------------------------
data = pd.concat([x,y],axis=1)
sns.heatmap(data.corr(),annot=True)
plt.show()
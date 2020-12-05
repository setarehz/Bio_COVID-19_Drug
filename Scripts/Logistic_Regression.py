import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from matplotlib import pyplot
from numpy import where
data1 = pd.read_csv(r'.\Data\clasification_input.csv')

all_cols = data1.columns.values
features_cols =['degree of ending node', 'max deg DTTT','Sum DTTT' ,'Max DTDT','Sum DTDT','max deg DDDT']
#Balanc the dataset
yek =data1[data1['Label']==1]
sefr =data1[data1['Label']==0]
print(len(sefr),'and',len(yek))
normalsefr=sefr.sample(n=1000)
dd=[normalsefr,yek]
data2=pd.concat(dd)
data2= pd.DataFrame(data2)
X = data2[features_cols] # Features
Y = data2['Label'] # Target variable

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
#Here, the Dataset is broken into two parts in a ratio of 70:30.

logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

#Model Evaluation using Confusion Matrix-------------------------
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

print("Accuracy LOGREG:",metrics.accuracy_score(y_test, y_pred))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cnf_matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cnf_matrix[i, j], ha='center', va='center', color='red')
plt.show()

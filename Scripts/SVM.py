import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split

data1 = pd.read_csv(r'.\Data\clasification_input.csv')
all_cols = data1.columns.values
features_cols =['degree of ending node', 'max deg DTTT','Sum DTTT' ,'Max DTDT','Sum DTDT','max deg DDDT']
yek =data1[data1['Label']==1]
sefr =data1[data1['Label']==0]
#print(len(sefr),'and',len(yek))
normalsefr=sefr.sample(n=1000)
dd=[normalsefr,yek]
data2=pd.concat(dd)
data2= pd.DataFrame(data2)
X = data2[features_cols] # Features
Y = data2['Label'] # Target variable

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.45,random_state=0)
#Here, the Dataset is broken into two parts in a ratio of 70:30.

clf = svm.SVC(decision_function_shape='ovo') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)


pca = PCA(n_components = 2)
X_train2 = pca.fit_transform(X_train)
clf.fit(X_train2, y_train)

yy =y_train.values
plot_decision_regions(X=X_train2,y=yy,clf=clf)

plt.xlabel(X.columns[0], size=10)
plt.ylabel(X.columns[1], size=5)
plt.title('SVM Decision Region Boundary', size=16)
plt.show()

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


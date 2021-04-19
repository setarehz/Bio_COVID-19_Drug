from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib as mpl
from numpy import *
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df_concat = pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\final_actualZ_ones_feature_s.csv')
# df_concat = pd.DataFrame(df)
df_concat=df_concat.dropna()

dataSefr=pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\New_unlabeled_after_two_cutoff.csv')
dataSefr = dataSefr.drop(dataSefr[dataSefr.Label == 1].index)


features1=['Max DDT', 'Sum DDT','max deg DDT','drug_centrality','Sum DTTT','Max DDDT','Sum DDDT','Max DTTT','Max DTDT','degree of starting node','degree of ending node','max deg DDDT','max deg DTTT','max deg DTDT','target_centrality','betweenness_cen drug','betweenness_cen target']
features=['Sum DDT', 'degree of ending node', 'max deg DTDT', 'Max DTDT', 'max deg DDDT', 'target_centrality']

X = df_concat[features]
y = df_concat['Label']
# print('before df_concat:',len(df_concat))
print("1")

#novel dataset
xx=dataSefr[features]
yy=dataSefr['Label']
print("2")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=175)
SVC_model = svm.SVC()
print("3")
KNN_model = KNeighborsClassifier(n_neighbors=2)
print("4")
# X_train=X_train.dropna()
# y_train=y_train.dropna()

print("5")
# SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
print("6")
# SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = KNN_model.predict(X_test)
print("7")
# Accuracy score is the simplest way to evaluate
# print('SVM accuracy rate:',accuracy_score(y_test,SVC_prediction))
# F1= f1_score(y_test, SVC_prediction,average='macro')
# print("SVM F1 rate:",F1*100,"%")
print('KNN accuracy rate:',accuracy_score(y_test,KNN_prediction))
F1= f1_score(y_test, KNN_prediction,average='macro')
print("KNN F1 rate:",F1*100,"%")
# But Confusion Matrix and Classification Report give more details about performance
# print('SVM matrix:',confusion_matrix(y_test,SVC_prediction))
print(classification_report(y_test,KNN_prediction))

print('------------------------------------------------------------------')
# SVC_prediction_novel = SVC_model.predict(xx)
KNN_prediction_novel = KNN_model.predict(xx)

# unique1, counts1 = np.unique(SVC_prediction_novel, return_counts=True)
# prediction1= dict(zip(unique1, counts1))
# print('SVM novel prediction:',prediction1)

unique2, counts2 = np.unique(KNN_prediction_novel, return_counts=True)
prediction2= dict(zip(unique2, counts2))
print('KNN novel prediction:',prediction2)

zero_predicted_index=[]
zero_pred_dataf=pd.DataFrame(KNN_prediction_novel)
for index, value in zero_pred_dataf.iterrows():
    if value.item()==1:
        zero_predicted_index.append(index)
        # print(dataSefr.iloc[index])

zero_pred_rows = dataSefr.iloc[zero_predicted_index]
zero_pred_rows=pd.DataFrame(zero_pred_rows)
zero_pred_rows.to_csv(r'C:\Users\setareh\Desktop\COVVV\novel_KNN2.csv', index=False, sep=',')

# print(len(zero_pred_rows))

#Add new predicted zeros to trainset
frames = [df_concat,zero_pred_rows]
df_concat1 = pd.concat(frames)
# df_concat1=df_concat.add(zero_pred_rows)
# df_concat=df_concat.dropna()
X = df_concat1[features]
y = df_concat1['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=15)

SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
SVC_prediction2 = SVC_model.predict(X_test)
KNN_prediction2 = KNN_model.predict(X_test)
print('second SVM accuracy rate:',accuracy_score(SVC_prediction2, y_test))
print('second KNN accuracy rate:',accuracy_score(KNN_prediction2, y_test))
print('Second SVM matrix:',confusion_matrix(SVC_prediction2, y_test))
print(classification_report(KNN_prediction2, y_test))

xx=dataSefr.drop(zero_predicted_index)
xx=xx[features]
#second novel prediction
SVC_prediction_novel2 = SVC_model.predict(xx)
unique4, counts4 = np.unique(SVC_prediction_novel2, return_counts=True)
prediction4= dict(zip(unique4, counts4))
print('SVM 2 novel prediction:',prediction4)


KNN_prediction3 = KNN_model.predict(xx)
unique3, counts3 = np.unique(KNN_prediction3, return_counts=True)
prediction3= dict(zip(unique3, counts3))
print('KNN 2 novel prediction:',prediction3)

# -------------------------------------------------------------------
xx=xx.drop(zero_predicted_index)
# third svm
zero_predicted_index=[]
zero_pred_dataf2=pd.DataFrame(SVC_prediction_novel2)
for index, value in zero_pred_dataf2.iterrows():
    if value.item()==-1:
        zero_predicted_index.append(index)
        # print(dataSefr.iloc[index])

zero_pred_rows2 = dataSefr.iloc[zero_predicted_index]
# print(len(zero_pred_rows))

#Add new predicted zeros to trainset
frames = [df_concat1,zero_pred_rows2]
df_concat2 = pd.concat(frames)
# df_concat1=df_concat.add(zero_pred_rows)
# df_concat=df_concat.dropna()
Xy = df_concat2[features]
yx = df_concat2['Label']

X_train, X_test, y_train, y_test = train_test_split(Xy, yx, test_size=0.30, random_state=15)

SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
SVC_prediction4 = SVC_model.predict(X_test)
KNN_prediction4 = KNN_model.predict(X_test)
print('3rd SVM accuracy rate:',accuracy_score(SVC_prediction4, y_test))
print('3rd KNN accuracy rate:',accuracy_score(KNN_prediction4, y_test))
print('3rd SVM matrix:',confusion_matrix(SVC_prediction4, y_test))
print(classification_report(KNN_prediction4, y_test))

xx=dataSefr.drop(zero_predicted_index)
xx=xx[features]
#second novel prediction
SVC_prediction_novel3 = SVC_model.predict(xx)
unique4, counts4 = np.unique(SVC_prediction_novel3, return_counts=True)
prediction4= dict(zip(unique4, counts4))
print('SVM 3 novel prediction:',prediction4)





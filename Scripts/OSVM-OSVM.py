from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib as mpl
from numpy import *
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.metrics import f1_score
import sklearn
from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import sys
from sklearn.svm import SVC

#import datasets
ones = pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\Final_just_ones.csv')

actual_zero=pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\final_Final_Actual_zeroes_after_cutoff.csv')
actual_zero=actual_zero.dropna()
actual_zero["Label"]=-1

zeros=pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\New_unlabeled_after_two_cutoff.csv')
zeros= zeros.drop(zeros[zeros.Label == 1].index)
zeros["Label"]=-1

features1=['Max DDT', 'Sum DDT','max deg DDT','drug_centrality','Sum DTTT','Max DDDT','Sum DDDT','Max DTTT','Max DTDT','degree of starting node','degree of ending node','max deg DDDT','max deg DTTT','max deg DTDT','target_centrality','betweenness_cen drug','betweenness_cen target']
features=['Sum DDT', 'degree of ending node', 'max deg DTDT', 'Max DTDT', 'max deg DDDT', 'target_centrality']

#OSCM 1
X=ones[features]
Y=ones['Label']

train, test = train_test_split(ones, test_size=.25,random_state=503642)
svm = OneClassSVM(kernel='rbf', gamma=0.0001)
classifier = svm.fit(train[features],train["Label"])

#predict for testset and actual zeros
frames = [test,actual_zero]
test_concat = pd.concat(frames)
y_pred = svm.predict(test_concat[features])

print("Actual Test set:",test_concat.Label.value_counts())

unique3, counts3 = np.unique(y_pred, return_counts=True)
prediction= dict(zip(unique3, counts3))
print('Prediction of Test set:',prediction)


accuracy_score= accuracy_score(test_concat["Label"], y_pred)
print("Accuracy rate:",accuracy_score*100,"%")

F1= f1_score(test_concat["Label"], y_pred,average='macro')
print("F1 rate:",F1*100,"%")
print('OSVM confusion matrix:')
print(confusion_matrix(test_concat["Label"],y_pred, labels=[-1,1]))

#Novel prediction
xx=zeros[features]
y_novel_pred=svm.predict(xx)
unique1, counts1 = np.unique(y_novel_pred, return_counts=True)
prediction1= dict(zip(unique1, counts1))
print('Novel Osvm1 found:',prediction1)



# ===================================================================
#Second OSVM

#novelrows Ones!
ones_predicted_index=[]
ones_pred_dataf=pd.DataFrame(y_novel_pred)

for index, value in ones_pred_dataf.iterrows():
    if value.item()==1:
        ones_predicted_index.append(index)
        # print(value)

ones_pred_rows = zeros.iloc[ones_predicted_index]
ones_pred_rows["Label"]=1
zeros=zeros.drop(ones_predicted_index)

ones_pred_rows=pd.DataFrame(ones_pred_rows)
ones_pred_rows.to_csv(r'C:\Users\setareh\Desktop\COVVV\novel_OSVM1.csv', index=False, sep=',')

framess = [ones,ones_pred_rows]
df_concat1 = pd.concat(framess)

train2, test2 = train_test_split(df_concat1, test_size=.25,random_state=503642)
svm = OneClassSVM(kernel='rbf', gamma=0.00001)
framesss1 = [train2,df_concat1]
train_concat_osvm2 = pd.concat(framesss1)
classifier = svm.fit(train_concat_osvm2[features],train_concat_osvm2["Label"])

#predict for testset and actual zeros
framesss = [test2,actual_zero]
test_concat_osvm2 = pd.concat(framesss)
y_pred_osvm2 = svm.predict(test_concat_osvm2[features])

print("Actual Test set for OSVM2:",test_concat_osvm2.Label.value_counts())

unique, counts = np.unique(y_pred_osvm2, return_counts=True)
prediction= dict(zip(unique, counts))
print('Prediction of Test set for OSVM2:',prediction)

F1= f1_score(test_concat_osvm2["Label"], y_pred_osvm2,average='macro')
print("F1 rate OSVM2:",F1*100,"%")
print('OSVM2 confusion matrix:')
print(confusion_matrix(test_concat_osvm2["Label"],y_pred_osvm2, labels=[-1,1]))


# accuracy_scoree= accuracy_score(y_true =test_concat_osvm2["Label"],y_pred= y_pred_osvm2["Label"])
# print("Accuracy rate OSVM2:",accuracy_scoree*100,"%")


#Novel prediction OSVM2
xxx=zeros[features]
y_novel_pred2=svm.predict(xxx)
unique12, counts12 = np.unique(y_novel_pred2, return_counts=True)
prediction12= dict(zip(unique12, counts12))
print('Novel Osvm2 found:',prediction12)
# =====================================================================
# Third OSVM

#novelrows Ones!
ones_predicted_index=[]
ones_pred_dataf2=pd.DataFrame(y_novel_pred2)

for index, value in ones_pred_dataf2.iterrows():
    if value.item()==1:
        ones_predicted_index.append(index)
        # print(value)

ones_pred_rows3 = zeros.iloc[ones_predicted_index]
ones_pred_rows3["Label"]=1
# xxx=xxx.drop(ones_predicted_index)

ones_pred_rows3=pd.DataFrame(ones_pred_rows3)
ones_pred_rows3.to_csv(r'C:\Users\setareh\Desktop\COVVV\novel_OSVM2.csv', index=False, sep=',')


framess = [ones,ones_pred_rows3,ones_pred_rows]
df_concat3 = pd.concat(framess)

train3, test3 = train_test_split(df_concat3, test_size=.25,random_state=503642)
svm = OneClassSVM(kernel='rbf', gamma=0.00001)
frames3 = [train3,df_concat3]
train_concat_osvm3 = pd.concat(frames3)
classifier = svm.fit(df_concat3[features],df_concat3["Label"])

#predict for testset and actual zeros
framesss = [test3,actual_zero]
test_concat_osvm3 = pd.concat(framesss)
y_pred_osvm3 = svm.predict(test_concat_osvm3[features])

print("Actual Test set for OSVM3:",test_concat_osvm3.Label.value_counts())

unique, counts = np.unique(y_pred_osvm3, return_counts=True)
prediction= dict(zip(unique, counts))
print('Prediction of Test set for OSVM3:',prediction)

F1= f1_score(test_concat_osvm3["Label"], y_pred_osvm3,average='macro')
print("F1 rate OSVM3:",F1*100,"%")
print('OSVM3 confusion matrix:')
print(confusion_matrix(test_concat_osvm3["Label"],y_pred_osvm3, labels=[-1,1]))

#Novel prediction OSVM2
xxxx=zeros[features]
y_novel_pred3=svm.predict(xxxx)
unique13, counts13 = np.unique(y_novel_pred3, return_counts=True)
prediction13= dict(zip(unique13, counts13))
print('Novel Osvm3 found:',prediction13)
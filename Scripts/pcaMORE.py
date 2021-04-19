from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib as mpl
from numpy import *
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.metrics import f1_score
import sklearn
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import sys
# np.set_printoptions(threshold=sys.maxsize)

df = pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\normalized_ones.csv')
df = pd.DataFrame(df)
# dataSefr=pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\random_pairssefr_normal.csv')
dataSefr=pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\All_zero_normal.csv')
actual_zero=pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\no_path_zeros_normal.csv')
actual_zero=actual_zero.dropna()
# dataSefr["Sum DTDT"] = pd.to_numeric(dataSefr["Sum DTDT"])
dataSefr = dataSefr.drop(dataSefr[dataSefr.Label == 1].index)
# dataSefr = dataSefr.drop('Sum DTDT', 1)
# print('ALLLL0_pair_summery:',dataSefr.dtypes)
# print('All0_pair_summery:',dataSefrrrr.dtypes)

features1=['Max DDT', 'Sum DDT','max deg DDT','drug_centrality','Sum DTTT','Max DDDT','Sum DDDT','Max DTTT','Max DTDT','degree of starting node','degree of ending node','max deg DDDT','max deg DTTT','max deg DTDT','target_centrality','betweenness_cen drug','betweenness_cen target']
features=['max deg DDDT','target_centrality','degree of starting node','degree of ending node','max deg DTDT']
X=df[features]
y=df['Label']



xx=dataSefr[features]
yy=dataSefr['Label']

xxx=actual_zero[features]
yyy=actual_zero['Label']

# df['Label'].replace(0, 'No',inplace=True)
# df['Label'].replace(1, 'Yes',inplace=True)


x = df.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features
xx = dataSefr.loc[:, features].values
xx = StandardScaler().fit_transform(xx) # normalizing the features

xxx = actual_zero.loc[:, features].values
xxx = StandardScaler().fit_transform(xxx) # normalizing the features

pca_model = PCA(n_components=3)
principalComponents_breast = pca_model.fit_transform(x)
principalComponents_novel = pca_model.fit_transform(xx)
principalComponents_actualz = pca_model.fit_transform(xxx)

principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal1', 'principal2','principal3'])
principal_breast_Df['Label']=df['Label']

principal_novel_Df = pd.DataFrame(data = principalComponents_novel
             , columns = ['principal1', 'principal2','principal3'])
principal_novel_Df['Label']=dataSefr['Label']

principal_actualz_Df = pd.DataFrame(data = principalComponents_actualz
             , columns = ['principal1', 'principal2','principal3'])
principal_actualz_Df['Label']=actual_zero['Label']
#Drop outliers
# for x in range(50):
#     principal_breast_Df = principal_breast_Df.drop(principal_breast_Df[principal_breast_Df.principal2 == principal_breast_Df.principal2.max()].index)

#
# for y in range(50):
#     principal_breast_Df = principal_breast_Df.drop(
#         principal_breast_Df[principal_breast_Df.principal1 == principal_breast_Df.principal1.max()].index)
#     # principal_breast_Df = principal_breast_Df.drop(
#     #     principal_breast_Df[principal_breast_Df.principal1 == principal_breast_Df.principal1.min()].index)

# z = np.array(principal_breast_Df.Label)
print('Entire ones dataset:',principal_breast_Df["Label"].value_counts())
# #print(principal_breast_Df.principal2.max())
# print('Explained variation per principal component: {}'.format(pca_model.explained_variance_ratio_))
# # col = principal_breast_Df.Label.map({0:'b', 1:'r'})
# cmap = plt.cm.coolwarm
# norm = plt.Normalize(vmin=0, vmax=1)
# # color = ['red' if z.any()==0  else 'blue']
# principal_breast_Df.plot.scatter(x="principal1", y="principal2", c = cmap(norm(z)))
# plt.show()


#classification
train, test = train_test_split(principal_breast_Df, test_size=.25,random_state=503642)
train_normal = train[train['Label']==1]
train_outliers = train[train['Label']==0]
outlier_prop = len(train_outliers) / len(train_normal)
svm = OneClassSVM(kernel='rbf', gamma=7.04967)
classifier = svm.fit(train_normal[["principal1","principal2","principal3"]])

# x = train['principal1']
# y = train['principal2']
# # plt.scatter(x, y, alpha=0.7, c=train['Label'])
# # plt.xlabel('principal1')
# # plt.ylabel('principal2')
# # plt.show()

#predict
# x = test['principal1']
# y = test['principal2']
print('Dataset Actual zeros:',principal_actualz_Df["Label"].value_counts())
frames = [test,principal_actualz_Df]
# test_concat = pd.concat(frames)
test_concat = test.add(principal_actualz_Df)
# test_concat =test_concat.dropna()
where_are_NaNs = isnan(test_concat)
test_concat[where_are_NaNs] = 0
y_test=test_concat["Label"]
y_pred = svm.predict(test_concat[['principal1','principal2','principal3']])
# colors = np.array(['#377eb8', '#ff7f00'])
# plt.scatter(x, y, alpha=0.75, c=colors[(y_pred + 1) // 2])
# plt.xlabel('principal1')
# plt.ylabel('principal2')
# plt.show()

#novel predict
# x_axis=principal_novel_Df['principal1']
# y_axis=principal_novel_Df['principal1']

y_novel_pred=svm.predict(principal_novel_Df[['principal1','principal2','principal3']])
# colors = np.array(['#377000', '#ff7fff'])
# plt.scatter(x_axis, y_axis, alpha=0.7, c=colors[(y_novel_pred + 1) // 2])
# plt.xlabel('principal1')
# plt.ylabel('principal2')
# plt.show()

uniquee, countss = np.unique(train.Label, return_counts=True)
trainn= dict(zip(uniquee, countss))
print("train set (fed into the model):",trainn)
#actual_yek = test.Label.groupby().count()
print('actual test set for accuracy',test_concat.Label.value_counts())

unique3, counts3 = np.unique(y_pred, return_counts=True)
prediction= dict(zip(unique3, counts3))
print('prediction of test set for calculating accuracy:',prediction)

actual_y_test = test_concat.Label
f1 = f1_score(actual_y_test, y_pred,average='macro')
print('F1 score:',f1)
actual_y_test=actual_y_test.dropna()
accuracy_score= accuracy_score(actual_y_test, y_pred)
print('Accuracy rate:',accuracy_score)
# confusion matrix
matrix = confusion_matrix(y_true=actual_y_test,y_pred=y_pred)
print('Confusion matrix : \n',matrix)
# print("actual: ",test.Label,"p:",y_pred)
unique1, counts1 = np.unique(y_novel_pred, return_counts=True)
prediction1= dict(zip(unique1, counts1))
print('novel found:',prediction1)

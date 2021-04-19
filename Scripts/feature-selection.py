import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.linear_model import RidgeCV, LassoCV
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import ExtraTreesClassifier

data = pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\final_actualZ_ones_feature_s.csv')

data.drop_duplicates(subset=['Drug_Protein Pair'], keep="first", inplace=True)
all_cols = data.columns.values


# drug_centrality has been deleted because has the least correlation with the label
# Max DDT', 'Sum DDT','max deg DDT have been deleted because DDDT has the same as DDT
# Sum DTDT has been deleted because Sum DTDT and Max DTDT have 0.86 relation so we delete one of them
#also it has the second least corelation with the label
# Sum DTTT has been deleted because Sum DTTT and Max DTTT have 0.83 relation so we delete one of them
#Also it has the third least corelation with the label

#delete 4taii ha and save 3taii ha

# data=data.drop(columns=['Max DDT', 'Sum DDT','max deg DDT','drug_centrality','Sum DTDT','Sum DTTT'], axis=1)
data=data.dropna()

features_cols =['Max DDT','Sum DDT','Max DDDT','Sum DDDT','Max DTTT','Sum DTTT','Max DTDT','Sum DTDT','degree of starting node','degree of ending node','max deg DDT','max deg DDDT','max deg DTTT','max deg DTDT','drug_centrality','target_centrality','betweenness_cen drug','betweenness_cen target']
X = data[features_cols] # Features
Y = data['Label'] # Target variable

#Using Pearson Correlation---1
plt.figure(figsize=(12,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Correlation with output variable
cor_target = abs(cor["Label"])
#Selecting highly correlated features
relevant_features = cor_target
print('relevant features:',relevant_features.sort_values())

#Recursive Feature Elimination--2
# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 6)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
print(features_cols)

# Embedded Method (feature selection using Lasso regularization)--3
reg = LassoCV()
reg.fit(X, Y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,Y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
print(imp_coef)
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")


#Backward Elimination---4
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(Y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>1):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print('backward elimination :',selected_features_BE)

# Feature Importance with Extra Trees Classifier
X = data.iloc[:,1:18]
Y = data.iloc[:,19]
# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
print(model.feature_importances_)
print(model.feature_importances_.sort_values())

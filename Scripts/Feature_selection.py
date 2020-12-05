from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.linear_model import RidgeCV, LassoCV
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv(r'.\Data\clasification_input.csv')
data = pd.DataFrame(data)

data.drop_duplicates(subset=['Drug_Protein Pair'], keep="first", inplace=True)
all_cols = data.columns.values
features_cols =['Max DDT', 'Sum DDT','Max DDDT','Sum DDDT','Max DDTT','Sum DDTT','Max DTTT','Sum DTTT','Max DTDT','Sum DTDT','degree of starting node','degree of ending node','max deg DDT','max deg DTT','max deg DDDT','max deg DDTT','max deg DTTT','max deg DTDT']
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
relevant_features = cor_target[cor_target>0.01]
print(relevant_features)

#Recursive Feature Elimination--2
# Feature extraction
model = LogisticRegression()
rfe = RFE(model, 6)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
print(features_cols)

#Embedded Method (feature selection using Lasso regularization)--3
reg = LassoCV()
reg.fit(X, Y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,Y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
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
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

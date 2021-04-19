import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
dd=[]
no=[]
df = pd.read_csv(r'C:\Users\setareh\Desktop\Proteinssss.csv')
weight = df['Weight']
# for w in weight:
#     if w>100 :
#         dd.append(w)
#

print(weight.describe())
mode=weight.mode().to_numpy()
std=weight.std()

d1 = pd.Series(weight)
d1.plot(kind='hist')
plt.show()

for index, row in df.iterrows():

    if row["interaction_source"] == "khorsand-gordon":
        no.append(row)


dff = pd.DataFrame(no,columns=['Human_P','Human_G','Human_length','Weight','virus_protein_source','interaction_source'])
# print(no)
# print(len(no))


ddd = pd.Series(dff["Weight"])
ddd.plot(kind='hist')
plt.show()

print(dff["Weight"].describe())
# print(dff["Weight"].sort_values())
plus = df["Weight"].max()-dff["Weight"].max()
plus=math.floor(plus)
print('ghabl',df.iloc[2428])
for index, row in df.iterrows():
     if row["interaction_source"] == "khorsand-gordon":
         #print("ghablesh",row["Weight"],row["interaction_source"],index)
         df.loc[index, 'Weight']=df.loc[index, 'Weight']+plus
         #print("badesh", row["Weight"], row["interaction_source"])



print('baWd',df.iloc[2428])
# # for d in dd :
# #     if d>120:
# #         print(d)
#
# df["new_score"] = (df["Similarity"]-df["Similarity"].mean()) / std
# d2 = pd.Series(df["new_score"])
# d2.plot(kind='hist')
# plt.show()
#Z-score =.(raw score -mean score) / std dev of the scores



print(df["Weight"].describe())
seri = pd.Series(df["Weight"])
seri.plot(kind='hist')
plt.show()
df.to_csv(r'C:\Users\setareh\Desktop\COVVV\PPI_setscoring_without_nsp.csv', index=False, sep=',')
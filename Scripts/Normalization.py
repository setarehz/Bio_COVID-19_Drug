import pandas as pd
#PPI scores normalization
# data = pd.read_csv(r'.\Data\PPI.csv')
from sklearn import preprocessing

#Weight = data[['Weight']]
#normalized_w=(Weight-Weight.min())/(Weight.max()-Weight.min())
#df = pd.DataFrame(data)
#df['Normalized_weight'] = normalized_w
#df.to_csv('final_PPI.csv', index=False, sep=',')
#print(df['Normalized_weight'])
#-------------------------------------------------------------------------#
#genes and drugs scores normalization
data = pd.read_csv(r'.\Data\Drugs and genes with scores.csv')

Query_score = data[['Query Score']]
Interaction_score = data[['Interaction Score']]

normalized_Q = (Query_score-Query_score.min())/(Query_score.max()-Query_score.min())
normalized_I = (Interaction_score-Interaction_score.min())/(Interaction_score.max()-Interaction_score.min())

df = pd.DataFrame(data)
df['Normalized_query_score'] = normalized_Q
df['Normalized_interaction_score'] = normalized_I

#Seprate the drug names and gene names into two coloumns
drugold = data['Drug']

drugnew = []
for each in drugold :
    drugnew.append(each.lstrip())
data['Drug_name']=drugnew

#save to a new CSV file
df.to_csv('ffinal_Drugs_genes.csv', index=False, sep=',')
#print(df['Normalized_query_score'].max())

from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import DataStructs
import pandas as pd


# construct the chemical reactions
df_3 = pd.read_csv(r'.\Data\no_dup_drugs.csv')
#print(df_3)
df_smiles = df_3['Smile']
c_smiles = []
df_names = df_3['drug']
cc_names=[]

for dss in df_names:
    try:

        cc_names.append(dss)
    except:
        print('Invalid Names:', dss)

for ds in df_smiles:
    try:
        cs = Chem.CanonSmiles(ds)
        c_smiles.append(cs)
    except:
        print('Invalid SMILES:', ds)
#print()
# make a list of mols
ms = [Chem.MolFromSmiles(x) for x in c_smiles]

# make a list of fingerprints (fp)
fps = [FingerprintMols.FingerprintMol(x) for x in ms]

# the list for the dataframe
qu, ta, sim= [], [], []
names = []
print(type(c_smiles))
print(type(cc_names))
# compare all fp pairwise without duplicates
for n in range(len(fps)-1): # -1 so the last fp will not be used
    s = DataStructs.BulkTanimotoSimilarity(fps[n], fps[n+1:]) # +1 compare with the next to the last fp
    #print(c_smiles[n], c_smiles[n+1:]) # witch mol is compared with what group
    # collect the SMILES and values
    for m in range(len(s)):
        qu.append(c_smiles[n])
        ta.append(c_smiles[n+1:][m])
        sim.append(s[m])
        names.append(cc_names[n] + ';' + cc_names[n+1:][m])

# build the dataframe and sort it
d = {'names' : names,'query':qu, 'target':ta, 'Similarity':sim}
df_final = pd.DataFrame(data=d)
df_final = df_final.sort_values('Similarity', ascending=False)
i = df_final[((df_final.Similarity == 1) | (df_final.Similarity == 0))].index
#delete similarities with 1 and 0 score
df_final=df_final.drop(i)


#Organized the drug similarities data
df_final[['First_Drug_Name','Second_Drug_Name']] = df_final.names.str.split(";",expand=True,)
df = pd.DataFrame(df_final)
cols = list(df)
#add two new col
cols.insert(0, cols.pop(cols.index('First_Drug_Name')))
cols.insert(1, cols.pop(cols.index('Second_Drug_Name')))
df = df.reindex(columns= cols)
df = df.drop(columns=['names'])
df = df.rename(columns = {"query":"First_Drug_SMILES"})
df = df.rename(columns = {"target":"Second_Drug_SMILES"})
df = df.rename(columns = {"Similarity":"Similarity_Score"})
# save as csv
df.to_csv('.\Data\Drug_similarities.csv', index=False, sep=',')


import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\Final_unlabeled.csv')

degree_end=df['degree of ending node']

degree_end_arr=[]
degree_end_index=[]

for index,value in degree_end.items():
    if value <= 0.015625:
        degree_end_arr.append(value)
        degree_end_index.append(index)

print(len(degree_end_arr))

target_cen=df['target_centrality']
target_arr=[]
target_index=[]
for index, value in target_cen.items():
    if value==0:
        target_arr.append(value)
        target_index.append(index)

print("Target_cen zero:",len(target_arr))

ciz=[]
for i in target_index:
    if i in degree_end_index:
      ciz.append(i)

print("Common:",len(ciz))

indexes=np.concatenate((target_index, degree_end_index), axis=None)
print("len of Concat indexes:",len(indexes))
new_actualzeros=df.iloc[indexes]
new_actualzeros=pd.DataFrame(new_actualzeros)
df1=df.drop(index=indexes,axis=1)

new_actualzeros.to_csv(r'C:\Users\setareh\Desktop\COVVV\New_Actual_zeros_from_two_cutoff.csv', index=False, sep=',')
df1.to_csv(r'C:\Users\setareh\Desktop\COVVV\New_unlabeled_after_two_cutoff.csv', index=False, sep=',')

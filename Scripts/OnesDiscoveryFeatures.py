import pandas as pd
import numpy as np


#import datasets
ac_zeros = pd.read_csv(r'C:\Users\setareh\Desktop\COVVV\Final_Actual_zeroes.csv')

# features=['max deg DDDT','target_centrality','degree of starting node','degree of ending node','max deg DTDT']

ac_zerosdf=pd.DataFrame(ac_zeros.describe())
print(ac_zeros.describe())
ac_zerosdf.to_csv(r'C:\Users\setareh\Desktop\COVVV\Describe_actual_zeros.csv', index=False, sep=',')

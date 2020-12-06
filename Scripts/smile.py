from urllib.request import urlopen
import pandas as pd


def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + ids + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return 'NULL'


drug = pd.read_excel('DGdrug.xlsx')
our_drugs = drug["drug"]


DGdrug=[] # putting the column into a list
for i in our_drugs:
    DGdrug.append(i)

i=1
for ids in DGdrug[1:1000] :
    drug.at[i, 'Smile'] = CIRconvert(ids) # ? takes a lot of time
    i+=1
drug.to_excel('DGdrug.xlsx') #saving it



# Graph Presentation

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pyvis.network import Network
import random
import xlsxwriter
from scipy.sparse import csr_matrix
from colour import Color



drug_gene_data = pd.read_csv('.\Data\final_Drugs_genes.csv') # importing Dataset
drugs = drug_gene_data["Drug_name"]
genes = drug_gene_data["GENE"]
interaction = drug_gene_data['Normalized_interaction_score']

#
# score = pd.read_excel('Query_Interaction.xlsx')
# interaction = score['Interaction Score']


def convert_column_to_list(x):# puting a column into a list style
    result = []
    for i in x:
        result.append(i)
    return result

#.....................................Print number of duplicate in a list
# duplicate_drugs =convert_column_to_list(drugs)
# import collections
# print(len([item for item, count in collections.Counter(duplicate_drugs).items() if count > 1]))
#............................................................................


# generate_graph_dic(x, y) is a function that takes two lists arguments and generate a dictionary with
# gene keys and a list of drugs as its value
def generate_graph_dic(x,y):
    Res = {}  # making a dic to translate uniport id to human gene
    for i in x:
        Res[i] = []
    for i in range(len(x)):
        a = Res[x[i]]
        a.append(y[i])
        Res[x[i]] = a
    return Res

# print(generate_graph_dic(['a','b','c','c'],['e','e','f','g']))
# generate_edges(graph_dictionary) :
# produce a list of tuples as our edges (Having a (graph)dictionary of related nodes)
def generate_edges(graph_dictionary):
    edges_list = []
    for key, val in graph_dictionary.items():  # Some of the values of the keys are list(the genes which have multiple interaction)
        if len(val)==1:  # if it has just one drug so we have an edge
            edges_list.append((key, val[0]))  # edges are like tuple with starting and ending node
        else:
            for ele in val:  # incase that there are more drugs
                edges_list.append((key, ele))
    return edges_list

# print(generate_edges(generate_graph_dic(['a','b','c','c'],['e','e','f','g'])))

#....................Function that gets a list and a color, returnes a list of tuples for node coloring
#For node attributes input is a list of nod and the color---> color
#For edge attributes input is a list of tuples(oure edges) and the color
def color_node_or_edge(list, color):
    colored_list = []
    for ele in list:
        colored_list.append((ele, {"color": color}))
    return colored_list


#.....................This function delete edges with 'Null'
def remove_null_edge(list_of_edge):
    res = []
    for ele in list_of_edge:
        if 'Null' in ele:
             res.append(ele) # putting all node element into rest list
    result = list(list(set(list_of_edge) - set(res))) # put differences between two list
    return result


def convert_tuple_dic(tup, di):
    di = dict(tup)
    return di

#....................................Negative DTI...............................
def get_random_combinations(x, y):
    result = []
    for i in range(100):
        x_rand_select = random.choice(x)
        y_rand_select = random.choice(y)
        tuple_rand = (x_rand_select, y_rand_select)
        result.append(tuple_rand)
    return result

# test = get_random_combinations(genes, drugs)
# print(test)
#.....................................Generating the Graph Drug-Gene ............................................
gene_nodes = convert_column_to_list(genes)# making a list of genes
drug_nodes= convert_column_to_list(drugs) # a list of drugs


#....................................DD Graph................................

drug_data = pd.read_csv('.\Data\Drugs Similarity.csv')

First_Drug = drug_data['First_Drug_Name']
Second_Drug = drug_data['Second_Drug_Name']
drug_drug_weight = drug_data['Similarity_Score']
#
First_Drug_list = convert_column_to_list(First_Drug)
Second_Drug_list = convert_column_to_list(Second_Drug)


#...................................PP Graph (Edges)..............................
PPI_data = pd.read_csv('.\Data\final_PPI.csv')

viruse_protein = PPI_data['Virus']
human_protein = PPI_data['Gene names']
human_protein_weight = PPI_data['Normalized_weight']

viruse_nodes = convert_column_to_list(PPI_data['Virus'])
human_nodes = convert_column_to_list(PPI_data['Gene names'])







protein_nodes = gene_nodes + viruse_nodes

#.................................Add weight
def add_score(df):
    records = df.to_records(index=False)
    result = list(records)
    return result


#......................................swape
# Swap tuple elements in list of tuples
# Using list comprehension
def swap_tuple_elements(list):
    res = [(sub[1], sub[0], sub[2]) for sub in list]
    return res

#........................................................................

#...........................................Graph..............................


G =nx.Graph()
G.add_nodes_from(gene_nodes )      # a list of nodes
G.add_nodes_from(drug_nodes)
G.add_nodes_from(viruse_nodes)

DPI_weighted_edge = add_score(drug_gene_data)
PDI_weighted_edge = swap_tuple_elements(DPI_weighted_edge)

DDI1_weighted_edge = add_score(drug_data)
DDI2_weighted_edge = swap_tuple_elements(DDI1_weighted_edge)

Human_Virus_weighted_edge = add_score(PPI_data)
Viruse_Human_weighted_edge = swap_tuple_elements(Human_Virus_weighted_edge)

G.add_weighted_edges_from(DPI_weighted_edge)
G.add_weighted_edges_from(PDI_weighted_edge)

G.add_weighted_edges_from(DDI1_weighted_edge)
G.add_weighted_edges_from(DDI2_weighted_edge)

G.add_weighted_edges_from(Human_Virus_weighted_edge)
G.add_weighted_edges_from(Viruse_Human_weighted_edge)
# graph_matrix= nx.adjacency_matrix(G) #adjacency matrix of DPI graph (or our 0,1)
# print(graph_matrix)




# drug_drug_weight_list = convert_column_to_list(drug_drug_weight)
# print('DDI_edge score=', add_score(DDI_edge, drug_drug_weight_list))


# print("Nodes of graph: ") # printing nodes and edges
# print(G.nodes())
# print("Edges of graph: ")
# print(G.edges())

#..................Illustration of the Graph...............

# nx.draw_spectral(G, with_labels=True,  font_weight='bold')
# plt.savefig("simple_path.png") # save as png
# plt.show() # display



#.......................Testing
# print(G.degree['PSMD11'] ) # print degree of a node
# print(list(G.adj['NORTRIPTYLINE'])) # print a list of neighbors of the node
#.......................................adjacency matrix DPI.........................................


MATRIX_DPI = nx.to_pandas_adjacency(G, dtype=int) #adjacency matrix in data frame work
# MATRIX_DPI.to_excel(r'C:\Users\pascal\anaconda3\PycharmProjects\pythonProject\bio\graph\Adjacency_Matrix.xlsx', index = False)

#............................Find Path......................................



def path_structure(G, drug, target):
    path_structure_c1 = [] # relationg to the D-D-T structure
    path_structure_c2 = [] # relationg to the D-T-T structure
    path_structure_c3 = [] # relationg to the D-D-D-T structure
    path_structure_c4 = [] # relationg to the D-D-T-T structure
    path_structure_c5 = [] # relationg to the D-T-T-T structure
    path_structure_c6 = [] # relationg to the D-T-D-T structure
    all_paths = list(nx.all_simple_paths(G, source=drug, target=target, cutoff=3))
    three_lenght=[]
    four_lenght = []

    for ele in all_paths:
        if len(ele)== 3:
            three_lenght.append(ele)# all path with lenght 3
        else:
            four_lenght.append(ele)# all path with lenght 4


    for ele in four_lenght:
        ele.remove(drug)
        ele.remove(target)

    for i in three_lenght:
            if i[1] in drug_nodes:
                path_structure_c1.append(i)
            else:
                path_structure_c2.append(i)
    for i in four_lenght:
        try:
            if i[0] in drug_nodes:
                if i[1] in drug_nodes:
                    path_structure_c3.append(i)
                if i[1] in protein_nodes:
                    path_structure_c4.append(i)
            if i[0] in protein_nodes:
                if i[1] in protein_nodes:
                    path_structure_c5.append(i)
                if i[1] in drug_nodes:
                    path_structure_c6.append(i)
        except (IndexError, ValueError):
            pass
        continue

    for i in range(len(path_structure_c3)):
        path_structure_c3[i]=[drug]+ path_structure_c3[i] +[target]
    for i in range(len(path_structure_c4)):
        path_structure_c4[i] = [drug] + path_structure_c4[i] + [target]
    for i in range(len(path_structure_c5)):
        path_structure_c5[i] = [drug] + path_structure_c5[i] + [target]
    for i in range(len(path_structure_c6)):
        path_structure_c6[i] = [drug] + path_structure_c6[i] + [target]
    res = [path_structure_c1, path_structure_c2, path_structure_c3, path_structure_c4, path_structure_c5, path_structure_c6]
    return res



# paths = list(nx.all_simple_paths(G, source='CARFILZOMIB', target='P0DTC1', cutoff=3))
# print('paths=', paths)
#path_structure(G, 'ANILINE', 'P0DTC6')


# ......................................................................................
def multiplyList(myList): # zarb adad ie list dr ham
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result

def path_product(G, path): # multiply weights of each edge in a path
    list=[G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1)]
    return multiplyList(list)


def feature_vector(G, drug, target):
    vector = []
    for i in path_structure(G, drug, target):
        if i == []:
            vector.append(0)
            vector.append(0)
        else:
            path_scores = []
            for ele in i:
                path_scores.append(path_product(G, ele))
            vector.append(max(path_scores))
            vector.append(sum(path_scores))
    return vector

# paths = list(nx.all_simple_paths(G, source='ANILINE', target='P0DTC6', cutoff=3))
# print(paths)
# print(path_structure(G, 'ANILINE', 'P0DTC6'))
# test = feature_vector(G, 'ANILINE', 'P0DTC6')
# print(test)

def lable(G, drug, target):
    if G.has_edge(drug, target):
        return 1
    else:
        return 0



def degree_feature(G, drug, protein):
    result = []
    result.append(G.degree[drug])
    result.append(G.degree(protein))
    path = path_structure(G, drug, protein)
    for i in path:
        if i==[]:
            result.append(0)
        else:
            each_structure = []
            for element in i:
                total = 0
                for j in element:
                    total += G.degree[j]
                each_structure.append(total)
            result.append(max(each_structure))
    return result




#..................................Feature File Generation..........................
unique_drug_nodes = list(dict.fromkeys(drug_nodes))
unique_protein_nodes = list(dict.fromkeys(gene_nodes))
columns_name =['Drug_Protein Pair', 'Max DDT', 'Sum DDT', 'Max DTT' , 'Sum DTT','Max DDDT',
               'Sum DDDT','Max DDTT','Sum DDTT',
               'Max DTTT', 'Sum DTTT','Max DTDT', 'Sum DTDT',
               'degree of starting node','degree of ending node','max deg DDT',
               'max deg DTT', 'max deg DDDT', 'max deg DDTT', 'max deg DTTT',
               'max deg DTDT', 'Label']


data_list = []
classification = pd.read_excel('.\Data\clasification.xlsx')

#these two loops generate a table of different pairs as well as their features and labels
for i in unique_drug_nodes[190:191]:
    for j in unique_protein_nodes[:]:
        row = []
        data = []

        row.append((i, j))
        row += feature_vector(G, i, j)
        row += degree_feature(G, i, j)
        row.append(lable(G, i, j))

        # data.append(columns_name)
        data.append(row)

        data_list.append(data[0])
        classification_length = len(classification)
        classification.loc[classification_length] = row

# res = pd.DataFrame(data_list, columns=columns_name)
# res.to_excel(r'C:\Users\pascal\anaconda3\PycharmProjects\pythonProject\bio\graph\finall\clasification.xlsx',
#                   index=False)


#
classification.to_excel(r'.\Data\clasification.xlsx',
                  index=False)


# Bio_COVID-19_Drug
Applying Graph Mining to Predict Drug-Protein Interaction: A Search for COVID-19 Drug


Whats the Problem ?
The problem can be illustrated as a heterogeneous graph. nodes can be drugs or proteins. The weighted edges between them clarify the similarity of a pair of drug-drug nodes and also the interaction between protein-protein and drug- protein target.
This is a link prediction problem which we are going to predict new edge with drug and protein node endings.  

Datasets:
First step is gathering the datasets from Gordon dataset in “A SARS-CoV-2 protein interaction map reveals targets for drug repurposing” and ‘Khorsand’ article “SARS-CoV-2-human protein-protein interaction network” then Uniport website was used to map the uniport ID mentioned in 'Khorsand' article to provide a link between two above datasets.
After that, we extracted related drugs and scores from the Drug Gene Interaction Database with JS script.

--We obtain SMILES structures from drug formula with the help of ' NCI ' (National Cancer Institute), then similarity between each pair of drugs calculated by their fingerprints with the TANIMOTO algorithm.Similarity scores are in a range of (1, 0). It is obvious that, 1(one) shows the similarity of a drug with itself;   and 0(zero) shows this pair of drugs has no similarity in TANIMOTO's system ;On the other hand, there isn't any edge between these two particular drugs.For accuracy and improve our process, we removed rows with 1 (one) or 0 (zero) similarity scores.you can see the codes as Drug_similarities.Py in Script folder.
All datasest would be found in Data folder.


Methods:
First, we used TANIMOTO algorithm based on their smiles with Rdkit to find the similarity between drugs, which we would explain it, in more detail in next slide. 
Then we import our data to a network with all properties.
The next step is to apply graph mining techniques like depth first search for each group of path structures to determine paths with drug and protein endings with maximum 3 length with scores.Finally, the new DTIs are predicted based on our produced features by our classification model.

The Graph:
Mainly, there are three different edges in the graph, which connect protein-protein-nodes, drug-drug nodes and protein-drug nodes. For each of these edges we have different scores.Notice that We just investigate 3 or less length path.  Due to paths structures we group and navigate paths with scores.we look at 6 different path structures. The main features are extracted from these structures. (DTT,DDT,DDDT,DTDT,DTTT,DDTT)
We shoud note that in we used Normalization for Drugs and genes interaction scores and path scores for each structures.

Features:
We have 20 features : 'Max DDT', 'Sum DDT','Max DTT' ,'Sum DTT','Max DDDT','Sum DDDT','Max DDTT','Sum DDTT','Max DTTT','Sum DTTT','Max DTDT','Sum DTDT','degree of starting node','degree of ending node','max deg DDT','max deg DTT','max deg DDDT','max deg DDTT','max deg DTTT','max deg DTDT'
By Max we mean the maximum score of a path for a specific path structure.
By Sum, we mean the summation of all paths for a specific path structure.
We also consider the degree of  start and end nodes and maximum degree for each path structure.
We used 4 different methods for feature selection such as: calculating all Pearson correlations, Recursive Feature Elimination, Embedded Method (feature selection using Lasso regularization) and  Backward Elimination.
After that we picked 6 most relevant and common features which determined by the above algorithms; 'degree of ending node', 'max deg DTTT','Sum DTTT' ,'Max DTDT','Sum DTDT','max deg DDDT' are chosen ones.

Classification:
We applied two different binary classifiers to predict novel DTIs, namely: logistic regression and Support vector machine (SVM) . In the first stage we divided our dataset into train and test sets by 70% and 30%.
Logistic regression is a proper algorithm for binary classification problems.you can find the code in Logistic_Regression.py in Scripts folder
Support Vector Machine (SVM) is a discriminating classifier.it determines the line separating the whole dataset in two parts where in each class lay in either side.Also, you can find the python cod of this method in Scripts folder.

Plots:
For increasing the undrestanding of models and confusion matrix we have plotted them in Plots folder.


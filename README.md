# Bio_COVID-19_Drug
Applying Graph Mining to Predict Drug-Protein Interaction: A Search for COVID-19 Drug


Whats the Problem ?
The problem can be illustrated as a heterogeneous graph. nodes can be drugs or proteins. The weighted edges between them clarify the similarity of a pair of drug-drug nodes and also the interaction between protein-protein and drug- protein target.
This is a link prediction problem which we are going to predict new edge with drug and protein node endings.  

Datasets:
We  used  three  main  datasets  in  this  survey:   Drug-Drug  Similarity  (DD  sim),  Protein-ProteinInteraction (PPI) and, Drug-Target Interaction (DTI) datasets.


Methods:
The problem of predicting novel DTIs can be addressed as a link prediction problem.  As mentionedin the previous section, we first constructed a weighted heterogeneous graph. So, the goal is inferringmissing links between drugs and targets.The weighted, undirected graphG(V,E) illustrated the dataset in which each edgee= (u,v)∈Ereferred to an interaction betweenuandvand a similarity if bothuandvwere drugs. 

Features:
The next step was generating features based on the paths between each pair of drug-target nodesin the graph.  ConsiderX={x1,x2,...,xn∗m}is the feature vector andY={y1,y2,...,yn∗m}isthe label vector wherenis the number of drugs andmis the number of targets.  For all possiblepairs  of  drug-targets,  we  have  a  feature  vector  and  a  label  which  is  1  if  there  exists  an  edgebetween  those  nodes  in  the  graph  and  0  otherwise.   Therefore,  there  is  a  classification  problemwhich  we  attempt  to  predict  novel  interactions  between  drugs  and  their  targets  using  differentreliable machine learning models.
we map the graphG(V,E) into spaceRdwhered <<|V|.  In other words, they represented each node in the graph by a feature vector that pre-served all properties of the main graph which was smaller than the real number of vertices in thegraph.Here, for each drug-target pair, we extracted 18 features.  The first 6 features were related tothe specific pair (namely, degree, Closeness Centrality, and Betweenness Centrality of both drugand protein nodes) and 12 remaining features were relevant to all paths length less than or equal3  between  them.   Closeness  centrality  and  Betweenness  Centrality  are  measures  that  show  the centrality  and  accessibility  of  a  node  in  a  given  graph.   Closeness  centrality  of  a  nodeuis  thereciprocal  of  the  average  shortest  path  distance  to  u  overalln−1  reachable  nodes.
whereσ(s,t) is the number of shortest (s,t)-paths,  andσ(s,t|v) is the weight of those pathspassing through some nodevother thans,t.  Ifs=t,σ(s,t) = 1, and ifv∈s,t,σ(s,t|v) = 0.Scores were calculated based on path scores used by DTiGEMS method.
In feature selection section, in the main dataset, we have 18 features and a label.  The label shows the known direct interactionbetween  a  drug  and  a  protein  target.   We  applied  five  different  methods  for  feature  selection:calculating all Pearson Correlations, Recursive Feature Elimination, Embedded Method (featureselection using Lasso Regularization),Backward Elimination and Extra Trees Classifier.After that we picked the 6 most relevant and common features determined by the above algo-rithms.
 
 Model:
 We  are  facing  a  high  imbalanced  dataset  containing  more  than  624,000  pairs  in  the  unlabeledclass (unknown interaction) and 764 pairs in the positive class (known interaction).  We createdthe negative class (without interaction) for classification and narrowing our results based on twoprinciples; first, we assumed unlabeled pairs which have no connecting path (between the drug andtarget) as a member of the negative class; comparing these pairs to the positive pairs by illustratingthe distribution of each features in the dataset, some noticeable results were observed; They werehighly in different in two features.  secondly, we assumed some other unlabeled pairs as negativepairs based on the detected differences.  At the end, we were able to split the data into 764 positivepairs, 543,325 negative pairs, and 80,965 unlabeled pairs.We  applied  different  models1to  predict  novel  DTIs;  six  models  were  chosen  based  on  theirreliability and F1 score which are described in the next subs-section.  OSVM, Isolation Forest, andEllipsis are mostly used for anomaly detection,  however,  when the dataset is highly imbalancedthey can be used to detect minority class or learn the pattern of just one class.
 Result:
In common novel DTIs predicted by the six models were stored; and repeated drugs were sortedbased on their frequency in the outcomes.  Finally, possible medicines for the COVID-19 predictedby the models and some publications proving the efficiency of those against the disease can be seen in result file in data folder.



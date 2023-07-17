import pandas as pd
import os
from collections import Counter
import networkx as nx
import numpy as np
import statsmodels.api as sm
import itertools
import matplotlib.pyplot as plt
import sys 
from networkx.algorithms import bipartite

from networkx.algorithms.centrality import *
sys.path.insert(1, '/home/azaiez/Documents/Cours/These/European Commission/Programs')
from classes.bigraph import *
from utils.load_data import *
path = '/home/azaiez/Documents/Cours/These/European Commission/'


## Load Data
meetings = pd.DataFrame() # dataframe containing all meetings of commissioners and their cabinet members
entities = pd.DataFrame() # dataframe containing all entities of 3 differents types : 'Commissioner','Cabinet Member', 'Organization'
# create a dataframe with commissioner schedule and 
meetings_path = path +'data/meetings/Commissioners/'
meetings , entities  = load_commissioner_schedules(meetings_path, meetings , entities )
# Create a data frame with cabinet schedule
meetings_path =  path +'data/meetings/Cabinets/'
meetings , entities  =  load_cabinet_member_schedules(meetings_path, meetings , entities )
# add organization to entities dataframe
entities = add_organizations(meetings, entities)
# remove organization S.A.
entities= entities[ entities['Name'] != 'S.A.']

## Transparency Regisgter
 #Transparency register 
orga_path = path
organisations = pd.read_excel(orga_path+ 'data/transparency_register.xls', engine='xlrd')

###################### Hypergraph Creation #####################

# Create a list of edges corresponding to entities present in mettings
hyperedges = get_hyperedges(meetings)

# Add entities as nodes to bipartite graph with corresponding attributes
B = nx.Graph()
for entity , function in zip (entities['Name'], entities['Type']):
        B.add_node(entity, bipartite= 'entity', Function = function )
# Add mettings to bipartite graph  
hyperedges_counts = Counter(hyperedges)
for i in range(len(hyperedges_counts)):
    B.add_node( i , bipartite= 'meeting')
# Add links between indivuduals and meetings    
for i , (hyperedge, count)  in enumerate( hyperedges_counts.items()):
    for entity in hyperedge:
        B.add_edge( i, entity , weigth = count )
        
nx.write_gexf(B, path+"/Programs/Bipartite.gexf")

## Centrality measures ##
hyperedges_carac = pd.DataFrame(bipartite.sets(B)[1] , columns =['ID'])

centrality_algo =[bipartite.betweenness_centrality_parallel,  bipartite.closeness_centrality]
centrality_label = ['Betweenness', 'Closeness']
for algo , label in zip (centrality_algo , centrality_label):
    centrality = algo(B, bipartite.sets(B)[0])
    entities[label] = [centrality[entity] for entity in entities['Name']]
    hyperedges_carac[label] =  [centrality[meeting] for meeting in hyperedges_carac['ID']]
    
#degree of nodes
degree = bipartite.degrees(B, bipartite.sets(B)[0])
hyperedges_carac['Cardinality'] = [degree[0][meeting] for meeting in hyperedges_carac['ID']]
entities['Degree'] = [degree[1][entity] for entity in entities['Name']]

#hyperedges_carac.to_csv(path + 'Programs/hyperedge_carac.csv')

#plot degrees hist

bins = np.arange(entities['Degree'].min(), entities['Degree'].max(), 20)
kwargs = {'alpha' : 0.5, 'bins' : bins, 'edgecolor' : 'k'}
plt.close()
types = [ 'Cabinet Member', 'Organizaion', 'Commissioner',]
for t in types :
    entities[ entities['Type'] == t]['Degree'].plot(kind ='hist', **kwargs, label =t)

plt.legend()
plt.xlabel('Degree of entities')
plt.yscale('log')
plt.show()

######################## Projected Graph ############################

def my_weight(G, u, v, weight="weight"):
    w = 0
    for nbr in set(G[u]) & set(G[v]):
        w += ( G[u][nbr].get(weight, 1) + G[v][nbr].get(weight, 1) ) /2
    return w
G = bipartite.generic_weighted_projected_graph( B, bipartite.sets(B)[0] , weight_function = my_weight)
G.remove_node('S.A.') # remove 'S.A.'
nx.write_gexf(G, path+"/Programs/Graph.gexf")

##Centrality Measures
centralities =[]
centralities.append(eigenvector_centrality(G, weight = "weight"))
centralities.append(G.degree(weight = 'weight'))
centralities.append(betweenness_centrality_parallel_unipartite(G))
labels = ['Eigenvector', 'Degree', 'Betweenness']


for centrality , label in zip (centralities, labels) :
    entities[label] = [centrality[entity] for entity in entities['Name']]


entities.to_csv(path +'Programs/entities_carac.csv')


##
plt.close(fig)
centralities = ['Eigenvector', 'Degree', 'Betweenness']
for centrality in centralities:
    entity_types = entities['Type'].unique()
    fig, ax = plt.subplots(1,3, sharey = True, sharex = True , figsize = (7,3))
    for i , entity_type in enumerate( entity_types ):
        entities[entities['Type'] == entity_type][centrality].plot(kind ='hist', ax =ax[i], alpha = 0.5, bins= 20)
        ax[i].set_title(f"{entity_type}")
        ax[i].set_yscale('log')
    fig.suptitle(centrality) 
    
    plt.tight_layout()
    fig.savefig(path + 'Programs/Figures/'+centrality+'.pdf')
plt.show()


## Restrict nodes set to entities appearing in the transparency register
# remove organisation that does not appear in the transparence resigiter
G = G.subgraph( set(entities['Name'] ) & set(organisations['Name']) | set(entities['Name'][entities['Type'] != 'Organisation']))

entities = entities[ entities['Name'].isin( G.nodes())]

def add_columns_from_orga_to_entities(df, orga_column_label, entities_column_label):
    dict = organisations.set_index('Name')[orga_column_label].to_dict() 
    df[entities_column_label] = [dict[orga] if typ == 'Organisation' else None  for orga, typ in zip(df['Name'], df['Type']) ]
    return(df)
    
#Add eu grant as columns in entities
entities = add_columns_from_orga_to_entities(entities,'Closed year EU grant: amount (source)' , 'E.U. Grants')
entities['E.U. Grants'] = entities['E.U. Grants'].fillna(0)
# Add other columns
column_labels  = [ 'Members FTE' , 'Form of the entity', 'Head office country', 'Interests represented', 'Level of interest']
for feature in column_labels :
    entities = add_columns_from_orga_to_entities(entities, feature, feature)


#correlation between E.U. grants and centralities

print(entities[entities['Type'] == 'Organisation' ] [['E.U. Grants', 'Members FTE', 'Eigenvector', 'Degree', 'Betweenness']].corr())

##
# ols regression
def ols_regression(df, y_col, x_cols):    # Extract the dependent variable (Y) and independent variables (X)
    Y = df[y_col]
    Y = pd.get_dummies(Y, drop_first=True)

    X = df[x_cols]    
    X = pd.get_dummies(X, drop_first=True)

    # Add a constant column to X for the intercept term
    X = sm.add_constant(X)

    # Perform OLS regression
    results = sm.OLS(Y, X).fit()

    return results
    
data  = entities[entities['Type'] == 'Organisation' ]
result = ols_regression(data, 'Head office country', [ 'Level of interest'])
print(result.summary())



##


def check_mixed_data_types(df):
    mixed_columns = []
    for column in df.columns:
        unique_types = df[column].apply(type).unique()
        if len(unique_types) > 1:
            mixed_columns.append(column)
    return mixed_columns







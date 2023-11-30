import pandas as pd
import os
from collections import Counter
import networkx as nx
import numpy as np
import statsmodels.api as sm
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from networkx.algorithms import bipartite

from networkx.algorithms.centrality import *
sys.path.insert(1, '/home/azaiez/Documents/Cours/These/European Commission/Programs')
from classes.bigraph import *
from classes.european_commission import *
from classes.orbis import *
from classes.transparency_register import *
from utils.centrality import *


path = '/home/azaiez/Documents/Cours/These/European Commission/'

#TODO Check the hyergraph creation. When restricting the maximal connected subgraph, a warning
#TODO Check the weights of the hyperedges




## Load Data
EU = EuropeanCommission()
meetings_path = path + 'data/meetings/'

EU.initialize( meetings_path)

# Transparency Register
TR = TransparencyRegister()
TR.initialize()

EU.add_data_to_ententies_from(TR.data, [ 'TR Country','Category of registration', 'Level of interest', 'Fields of interest', 'Lobbying cost' , 'Members FTE'] )
EU.entities['Category of registration'] = EU.entities['Category of registration'].str.replace('&' ,'and')

# Orbis
orbis = Orbis()
orbis.initialize()

df = orbis.matched_names.merge(orbis.company_data, left_on = 'BvD ID', right_on = 'BvD ID', how = 'right')

EU.add_data_to_ententies_from(df, right_on = 'Company name', columns = [ 'Orbis Country', 'Country ISO', 'City', 'BvD ID','BvD sectors', 'NAICS', 'Last year', 'Revenue', 'Shareholders funds',
       'Assets', 'Nb employees', 'corporate group size', 'Entity type',
       'GUO Name','GUO Country ISO', 'GUO Type', 'GUO NAICS', 'GUO Revenue', 'GUO Assets',
       'GUO Nb employees'])


#Centralities

centralities = pd.read_csv (path +'Programs/entities_centrality.csv')
columns  = ['Betweenness', 'Closeness', 'Degree','Strength']
EU.add_data_to_ententies_from(centralities,columns)


## Factor analysis of mixed data
import prince
import altair as alt

df = EU.get_companies()[['Betweenness', 'Closeness', 'Degree','Strength', 'TR Country','Members FTE', 'BvD sectors', 'Revenue', 'Assets', 'Nb employees', 'Level of interest', 'Fields of interest' ]].dropna().copy()


cols = [(col , col) for col in df.columns]
df.columns = pd.MultiIndex.from_tuples(cols)

#Split group data
groups = ['Level of interest', 'Fields of interest']
for group in groups :
    splited_df = split_column(df[group], group)
    cols = [(group, col) for col in splited_df.columns]
    splited_df.columns = pd.MultiIndex.from_tuples(cols)

    df.drop((group,group), axis = 1, inplace = True)
    df = pd.concat( [df , splited_df], axis = 1)
groups

mfa = prince.MFA(
    n_components=2,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=42
)
mfa = mfa.fit(df , groups = df.columns.levels[0].tolist())

print(famd.eigenvalues_summary, '\n\n')

print(famd.column_coordinates_,'\n\n')


print(famd.column_contributions_ )


famd.plot(
    dataset,
    x_component=0,
    y_component=1
)
alt.renderers.enable('altair_viewer')
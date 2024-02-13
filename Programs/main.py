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
import json
import numpy as np

sys.path.insert(1, '/home/azaiez/Documents/Cours/These/European Commission/Programs')
from classes.european_commission import *
from classes.orbis import *
from classes.transparency_register import *
from utils.centrality import *
from utils.core import *
from utils.ols import *
from sklearn.metrics import normalized_mutual_info_score

path = '/home/azaiez/Documents/Cours/These/European Commission/'


#TODO Check the hyergraph creation. When restricting the maximal connected subgraph, a warning
#TODO Check the weights of the hyperedges



## Load Data
meetings_path = path + 'data/meetings/'
EC = EuropeanCommission(path + 'data/')

# EU member

df = EC.EC_members_info()
titles = {'Cabinet member':'Cabinet member',
            'Commissioner' : 'Commissioner', 
            'Vice-President' : 'Commissioner', 
            'President' : 'Commissioner',
            'Executive Vice-President' : 'Commissioner', 
            'High Representative' : 'Commissioner',
            'Director-General': 'Director-General', 
            'Head of service' : 'Director-General', 
            'Acting Director-General' : 'Director-General',
            'Secretary-General' : 'Director-General', 
            'Acting Head of service' : 'Director-General',
            'Director of Office' : 'Director-General'}
            
df['Category of registration']  = df['Title'].map(titles) 

# # EC members that have changed their title ex : cabinet merber to DG
# df.drop_duplicates(['Name', 'Category of registration' ])[df.drop_duplicates(['Name', 'Category of registration' ]).duplicated('Name', keep = False)]

df = df.drop_duplicates('Name')
df.index = df['Name']
EC.add_data_to_entities_from(df, columns = 'Category of registration')

#.rename(columns = {'Type': 'Category of registration'}).drop_duplicates('Name')
# EU.add_data_to_entities_from(df, 'Category of registration')


# ######### Transparency Register ##########
TR = TransparencyRegister(path+ 'data/')

EC.add_data_to_entities_from(TR, [ 'TR Name','TR Country','Category of registration', 'Level of interest', 'Fields of interest',  'Members FTE'] )
EC.entities['Category of registration'] = EC.entities['Category of registration'].str.replace('&' ,'and')

def mapper_duplicates(df, duplicated_key):
    df = df.dropna(subset = duplicated_key)[df.dropna(subset = duplicated_key).duplicated(duplicated_key, keep = False)].copy()
    mapper = {}
    for name in df[duplicated_key]:
        TR_ids = df[df[duplicated_key] == name].index
        TR_ids = list(TR_ids)
        ref_key = list( set(EC.entities.index) & set(TR_ids))[-1]
        other_keys = list( set(TR_ids) - set(ref_key))
        mapper.update(dict( zip(other_keys , [ref_key]*len(other_keys)) ))
    return(mapper)        
        
        
duplicated_key = 'TR Name'
mapper = mapper_duplicates(EC.entities, duplicated_key)
EC.collapse_entities(mapper)


# ######## Orbis ##########
orbis = Orbis()
orbis.matched_names = orbis.matched_names.loc[ list( set(EC.entities.index) & set(orbis.matched_names.index))]

# Collapse entities with similar BvD ID

duplicated_key = 'BvD ID'
mapper = mapper_duplicates(orbis.matched_names, duplicated_key)
EC.collapse_entities(mapper)

# Add orbis data to entities
orbis.matched_names['TR ID'] = orbis.matched_names.index
df = orbis.matched_names.merge(orbis.company_data, left_on = 'BvD ID', right_on = 'BvD ID', how = 'right')

df.index = df['TR ID']

columns = [ 'Orbis Country', 'Country ISO',  'BvD sectors','Revenue', 
       'Assets', 'Nb employees', 'NACE', 'NACE core','corporate group size', 'Entity type',
       'GUO Name','GUO Country ISO', 'GUO Type', 'GUO NACE core', 'GUO Revenue', 'GUO Assets',
       'GUO Nb employees']
       
EC.add_data_to_entities_from(df,columns)

       
# ########### Centralities ##########

centralities = pd.read_csv (path +'Programs/node_centrality.csv', index_col = 'Name' )
columns  =centralities.columns
EC.add_data_to_entities_from(centralities,columns)

## 
with open( path + 'Programs/core.json') as json_file:
    core = json.load(json_file )
    
core = { int(m) : {int(k) :  [set(component) for component in core[m][k]] for k, value in core[m].items() } for m in core.keys()}

# with open( path + 'Programs/intimate_core.json') as json_file:
#     intimate_core = json.load(json_file )
#     
# intimate_core = { int(m) : {int(k) :  [set(component) for component in intimate_core[m][k]] for k, value in intimate_core[m].items() } for m in intimate_core.keys()}


##################### Save file for regression #################
df = EC.get_companies()[[ 'TR Country','Members FTE', 'Revenue', 'Level of interest', 'NACE']].dropna().copy()

#Split catagorial data

levels = split_column(df, 'Level of interest', prefix = 'Level')
df.drop('Level of interest', axis = 1, inplace = True)
# fields = split_column(df , 'Fields of interest', prefix = 'Field')
# df.drop('Fields of interest', axis = 1, inplace = True)

df = pd.concat( [df , levels], axis = 1)

df.to_csv(path + 'Programs/data_test_R.csv', index = False)

#fields.to_csv(path +'Programs/fields.csv', index = False)

###################### Computations #####################


attr = dict( zip (EC.entities.index , EC.entities['Type'] ))
B = XGI_2_nxBipartite(EC.H)
nx.set_node_attributes(B,attr,'Type')

nx.write_gexf(B, path+"/Programs/Bipartite.gexf")


## Centrality measures ##
centralities = [degree, strength, MSA , betweenness]
node_cent_df = pd.DataFrame(index = {'Name' : list(EC.H.nodes)})
edge_cent_df = pd.DataFrame(index = {'Name' : list(EC.H.edges)})

for algo in centralities:
    print(algo)
    cent = algo(EC.H)
    #cent = cent /np.linalg.norm(cent)
    label = [cent.columns[0] ]
    node_cent_df.loc[list(EC.H.nodes) ,  label]   = cent.loc[list(EC.H.nodes)]
    try :
        edge_cent_df.loc[list(EC.H.edges), label ] = cent.loc[list(EC.H.edges)]
    except:
        pass
        
# EV centrality    
for mode in ['linear', 'log exp', 'max']:
    cent = eigenvector(EC.H, mode)
    label = [cent.columns[0] ]

    node_cent_df.loc[list(EC.H.nodes) , label]   = cent.loc[list(EC.H.nodes)]
    edge_cent_df.loc[list(EC.H.edges), label ] = cent.loc[list(EC.H.edges)]  
    
# Cardinality of edges    
cent = cardinality(EC.H)
label = [cent.columns[0] ]
edge_cent_df.loc[list(EC.H.edges), label ] = cent.loc[list(EC.H.edges)]  
    
# Core decomposisition and Hypercoreness
core = core_decomposition(EC.H)
core = { int(m) : {int(k) :  [set(component) for component in core[m][k]] for k, value in core[m].items() } for m in core.keys()}

cent  = hypercoreness(core) 
label = [cent.columns[0] ]
node_cent_df.loc[list(EC.H.nodes) , label]   = cent.loc[list(EC.H.nodes)]


## Save results
# edge_cent_df.to_csv(path + 'Programs/edge_centrality.csv')
# node_cent_df.to_csv(path + 'Programs/node_centrality.csv')
# 
# # Convert sets to lists before serialization
# def convert_sets_to_lists(obj):
#     if isinstance(obj, set):
#         return list(obj)
#     return obj
# with open(path + "Programs/core.json", "w") as outfile: 
#     json.dump(core, outfile, default=convert_sets_to_lists)
#################### Vizualisation Results#####################
##  Describe nodes 

# Count
df_c = EC.entities[['Type','Category of registration']].groupby('Type').value_counts(dropna = False).copy()

# Strength
df_s = EC.entities.groupby(['Type','Category of registration'], dropna = False)['Strength'].sum().astype(int)


df = pd.concat([df_c, df_s]  ,axis = 1, join = 'inner')

df['mean strength'] = df['Strength']/ df['count']

print(df)
print( df.to_latex( float_format="%.2f"))

##    Pie plot distribution strength 
try:
    plt.close(fig)
except:
    pass

def dist_strength_cat(carac, threshold):
    # Calculate the sum of strengths for each category
    category_sum = EC.entities.groupby(carac)['Strength'].sum().astype(int).sort_values()
    
    # Calculate the percentage of each category
    category_percentage = category_sum / category_sum.sum()
    
    # Identify categories with less than 5% and group them into 'other'
    category_sum.index = category_sum.index.where( category_percentage > threshold, 'Others')
    category_sum = category_sum.groupby(carac).sum().sort_values()
    return(category_sum)

category_sum = dist_strength_cat('Category of registration', 0.015)
# Rename long label
category_sum.index = category_sum.index.str.replace('Non-governmental organisations, platforms and networks and similar',  'NGOs')
category_sum.index = category_sum.index.str.replace('Trade unions and professional associations', 'Trade Unions')
# Colors for piece of pies
ec_members = ['Commissioner', 'Cabinet member', 'Director-General' ]
orga = [ element for element in set(category_sum.index ) - set(ec_members) ]
cat_colors = {member : color for member , color in zip ( ec_members,  sns.color_palette("Oranges",5))}
cat_colors.update( {orga : color for orga, color in zip ( orga , sns.color_palette('bwr', 12))} )

colors = [cat_colors[cat] for cat in category_sum.index ]

fig, ax = plt.subplots()
wedges, texts, autotexts  = ax.pie(category_sum.values, labels=category_sum.index, autopct='%1.1f%%', startangle=120, colors = colors, textprops=dict(color="black", bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')), pctdistance=0.8, radius=1.2)
ax.axis('equal') 
fig.suptitle('Distribution of Strength', fontsize = 30, y = 0.9)
plt.setp(autotexts, size=10, weight="bold")

plt.tight_layout()
plt.show()


##
try:
    plt.close(fig)
except:
    pass
category_sum = dist_strength_cat('TR Country', 0.02)

# Colors for piece of pies
non_EC = ['UNITED STATES', 'UNITED KINGDOM', 'SWITZERLAND' ]
orga = [ element for element in set(category_sum.index ) - set(non_EC) ]
cat_colors = {member : color for member , color in zip ( non_EC,  sns.color_palette("RdPu",5))}
cat_colors.update( {orga : color for orga, color in zip ( orga , sns.color_palette('Blues', len(orga) , desat = 0.5))} )

colors = [cat_colors[cat] for cat in category_sum.index ]

fig, ax = plt.subplots()
wedges, texts, autotexts  = ax.pie(category_sum.values, labels=category_sum.index, autopct='%1.1f%%', startangle=120, colors = colors, textprops=dict(color="black", bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')), pctdistance=0.8, radius=1.2)
ax.axis('equal') 
#fig.suptitle('Distribution of Strength', fontsize = 30, y = 0.9)
plt.setp(autotexts, size=10, weight="bold")

plt.tight_layout()
plt.show()
##
#plot degrees hist
plt.close()
feature = 'Hypercoreness'
bins = np.arange(EC.entities[feature].min(), EC.entities[feature].max(), 1)
kwargs = {'alpha' : 0.5, 'edgecolor' : 'k' , 'bins' : bins}
plt.close()
types = EC.entities['Type'].unique()
for t in types :
    data = EC.entities[ EC.entities['Type'] == t][feature]
    sns.histplot(data , stat ='probability', **kwargs, label =t)

plt.legend()
plt.xlabel(feature)
plt.yscale('log')
plt.show()


## Correlation of centralities

columns = ['Degree', 'Strength', 'Betweenness', 'EV_linear', 'EV_log exp', 'EV_max', 'Hypercoreness']
heatmap_corr (EC.entities , columns    , annot = True)




##### Mean groupby category
fields = split_column(EC.get_orga() , 'Fields of interest')
y ='Strength'
x = fields
x_label = 'Fields of interest'

def _groupby(x , feature):
    return( pd.concat( [x , EC.entities] , axis =1).groupby(feature, dropna = False)[y])

sum_groupby = { feature : _groupby(x , feature).sum()[1] for feature in x}
std_groupby = { feature  : _groupby(x , feature).std()[1] for feature in x}
sem_groupby = { feature  : _groupby(x , feature).sem()[1] for feature in x}

print(pd.DataFrame({ x_label: sum_groupby.keys(), 'Sum': sum_groupby.values() ,  'Std' : std_groupby.values() , 'Sem' : sem_groupby.values()}).sort_values(by= 'Sum' , ascending=False))

df = EC.entities.groupby('Category of registration', dropna = False)[y]
#float_format="{:.2e}".format

## Exclusive categories
y = 'Strength'
x = 'TR Country'


#df = EC.get_orga()[EC.get_orga()['Category of registration']== 'Companies and groups'].groupby(x)[y]
df = EC.entities.groupby(x)[y]

print(df.sum().sort_values(ascending = False).astype(float).head(50))


plt.close()
plt.figure(figsize = (15, 7))
df = df.sum().sort_values(ascending = False)
plt.scatter(df.index , list(df), alpha = 0.7)
plt.xlabel('Country')
plt.ylabel('Total Strength')

plt.xticks(rotation=90)
plt.yscale('log')
plt.tight_layout()

plt.show()
##
df = pd.concat([EC.entities , levels], axis = 1)
df[ (df ['TR Country'] == 'BELGIUM') & (df['European'] == 1 )].groupby('Category of registration')[y].sum()


## Correlation btw hyperedge carac
edge_centrality  = pd.read_csv(path + 'Programs/edge_centrality.csv', index_col = 'Name')
edge_centrality['Cardinality'].mean()
edge_centrality['Cardinality'].std()
heatmap_corr (edge_centrality, edge_centrality.columns    , annot = True)


## Cumulative ditribution fonction of centralities according to category of registration
def cdf (ax, data, label):
    N=len(data)
    # sort the data in ascending order
    x = np.sort(data)
    # get the cdf values of y
    y = 1 - (np.arange(N) / float(N))
    # plotting
    ax.set_xlabel(label)
    ax.set_ylabel('CDF of %s' %label.lower())
    ax.plot(x, y, marker='o', markersize = 1.5, linewidth =0, alpha = 0.2)
    #ax.title('cdf of %s in log log scale' %label)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return(ax)
    
    
# feature  = 'Hypercoreness'
# try:
#      plt.close(fig)
# except:
#     None
# fig, ax = plt.subplots( figsize = (5,5))
# data = EC.entities[[ feature, 'Category of registration']].dropna()    
# categories = data['Category of registration'].unique()
# colors = sns.color_palette("hls", len(categories))
# for category , color in zip (categories , colors):
#     ax = cdf (ax, data[ data['Category of registration'] == category][feature] , feature, category, color)
#     ax.legend()
# fig.show()
# 


features = [EC.entities['Degree'], 
    EC.entities['Strength'], 
    cardinality(EC.H)['Cardinality'] ]
    
labels = ['Node degree', 'Node strength', 'Edge cardinality']
notations =['k_i', 's_i', 'd_e']

for feature,label,notation in zip(features,labels, notations):
    fig , ax = plt.subplots(figsize = (3,3))
    ax = cdf(ax,feature, notation, )
    plt.tight_layout()
    plt.savefig(path +f'Programs/Figures/{label}.pdf' )
    plt.close(fig)
    
    
##
df = EC.entities[EC.entities['Category of registration' ]== 'Companies and groups'].copy()
df_max = EC.entities[EC.entities['Category of registration' ]== 'Companies and groups'].copy()

def calculate_revenue_bis(row):

    if pd.notna(row['GUO Revenue']):
        return 1e3*row['GUO Revenue']
    elif pd.notna(row['Revenue']):       
        return row['Revenue']
    else:
        return np.nan

# Apply the custom function to create the 'revenue_bis' column
df_max['revenue_bis'] = df.apply(calculate_revenue_bis, axis=1)

# Print the updated DataFrame
print(df)
#plt.close(fig)

fig, ax = plt.subplots( 1,2, figsize = (10, 5))
ax[0].scatter( list(df_max['revenue_bis']) , list(df_max['Strength']))
ax[0].set_yscale('log')
ax[0].set_xscale('log')
ax[1].scatter( list(df_max['revenue_bis']) , list(df_max['Strength']))



plt.show()

##
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = EC.entities[EC.entities['Category of registration' ]== 'Companies and groups'].copy()
df.loc[df['Revenue'] == 0 , 'Revenue'] = np.nan
df.loc[df['GUO Revenue'] == 0 , 'GUO Revenue'] = np.nan



df['revenue_bis'] = df.apply(calculate_revenue_bis, axis=1)

df.dropna(subset = 'revenue_bis', inplace = True)    
df.dropna(subset = 'Strength', inplace = True)    

#df = df[ df['Strength'] > 20]

def linear_model(x, a, b):
    return a * x + b


x_data = np.log(list(df['revenue_bis']) )
y_data = np.log(list(df['Strength']))

# Fit the data to the model
bounds = ([1, -np.inf] , [np.inf, np.inf])
params, covariance = curve_fit(linear_model, x_data, y_data, bounds = bounds)

# params contains the optimized parameters (in this case, 'a' and 'b')
a, b = params

# Plot the original data
plt.scatter(x_data, y_data, label='Data', alpha = 0.5)

# Plot the fitted model
fitted_y = linear_model(x_data, a, b)
plt.plot(x_data, fitted_y, label='Fitted Line', color='red')

plt.legend()
plt.show()

###### Hypercore decomposition

msa = MSA(EC.H)
s = degree(EC.H)
##
m = 4
k = 6



plt.close()
df = pd.concat([msa, s], axis = 1)

fig, ax = plt.subplots(1,3, figsize = (14,3.5))

for i , (k,m) in enumerate([(0,0), (10,2) , (10,3)]):
    if i != 0 :
        df.loc[list(core[m][k][0] & set(EC.H.nodes))].plot('MSA', 'Degree', kind = 'scatter', ax = ax[i] , label = 'm = %d \n k = %d'%(m,k), color = 'k', alpha = 0.5)
    
    df.loc[EC.entities['Type'] == 'Organization'].plot('MSA', 'Degree', kind = 'scatter', alpha = 0.2, ax= ax[i], label = 'Organizations', color = (21/255,54/255,255/255))
    
    df.loc[EC.entities['Type'] == 'EC member'].plot('MSA', 'Degree', kind = 'scatter', alpha = 0.2, ax= ax[i], label = 'EC members', color = 'C1')

    ax[i].set_yscale('log')
    ax[i].set_xscale('log')

fig.suptitle('Core decomposition')
plt.legend()
plt.show()


## Heat map of n_k_m
#core_connectedness = proportion_largest_cc_size(core)
#core_connectedness = reverse_y_axis(core_connectedness)

n_k_m = k_m_core_size(core)/ EC.H.num_nodes
n_k_m = reverse_y_axis(n_k_m)

plt.close()
#cmap = plt.cm.gnuplot2.reversed()
cmap = plt.cm.plasma.reversed()
sns.heatmap(n_k_m, cmap =cmap, xticklabels = core[2].keys(), yticklabels = sorted(core.keys(), reverse = True))

plt.xlabel('k')
plt.ylabel('m')
plt.title (r'$ \frac{N_1^{k,m}}{N_1^{k,m} +N_2^{k,m}} $')
plt.title (r'$ n_{k,m}$')

#plt.savefig(path + '/Programs/Figures/n_k_m_heatmap_orga.pdf')
plt.show()
## df n_k_m
n_k_m = k_m_core_size(core)/ EC.H.num_nodes

df_n_k_m = pd.DataFrame( columns = ['m', 'k','n_k_m' ] )
for i, m in enumerate(core.keys()):
    for j, k in enumerate(core[2].keys()):
        df_n_k_m = pd.concat([df_n_k_m , pd.DataFrame ( {'m':[m], 'k':[k], 'n_k_m': [n_k_m[i][j]] }) ])

df_n_k_m



## mututal information between cores
df_mi = pd.DataFrame(columns = ['k1', 'k2', 'm1', 'm2' , 'mi'])
for m1 in core.keys():
    for m2 in core.keys():
        for k2 in core[2].keys():
            for k1 in core[2].keys():
                if k1>k2 and m1<m2:
                    try:    
                        df = pd.DataFrame()
                        df.index = EC.H.nodes()
                        
                        df['km'] = 0
                        
                        df.loc[list(core[m1][k1][0]), 'km'] =1
                        
                        df['mk'] = 0
                        df.loc[list(core[m2][k2][0]), 'mk'] =1
                        
                        mi = normalized_mutual_info_score(df['km'], df['mk'])
                        
                        df_mi = pd.concat([ df_mi, pd.DataFrame({'k1': [k1], 'k2': [k2], 'm1' : [m1], 'm2': [m2] , 'mi': [mi]}) ])
                    except:
                        pass




### n_k_m vs k or vs m
plt.close()

core_ =core
K = max(set(list (itertools.chain.from_iterable([ core_[i].keys() for i in core_.keys()]))))
M = max(core_.keys())

ms = [2 ,3, 4]
n_k_m = k_m_core_size(core_)

# for m in ms:
#     plt.plot( list(range(1,K+1)) , n_k_m[m-2, :], label = 'm=%d'%m, )
# plt.xlabel('k')

ks = [1,2,3]
for k in ks:
    plt.plot( list(range(2, M+1)) , n_k_m[:, k-1], label = 'k=%d'%k, )
plt.xlabel('m')

plt.ylabel (r'$ n_{k,m}$')

# plt.xticks(list(range(1,K+1)))
# plt.yticks( np.arange(0,1,0.1))
plt.legend()
plt.show()

## Survival spicies
plt.close()
EC_members  = set(EC.entities[ EC.entities['Type'] == 'EC member'].index)
companies =  set(EC.entities[ EC.entities['Category of registration'] == 'Companies and groups'].index)

node_sets= {'EC members' : set(EC.entities[ EC.entities['Type'] == 'EC member'].index),
            'Companies' : set(EC.entities[ EC.entities['Category of registration'] == 'Companies and groups'].index),
            'Trade and business associations' : set(EC.entities[ EC.entities['Category of registration'] == 'Trade and business associations'].index),
            'NGO': set(EC.entities[ EC.entities['Category of registration'] == 'Non-governmental organisations, platforms and networks and similar'].index)
            }

m = 5
for label , node_set in node_sets.items():
    p = {}
    for k in core_[m].keys(): 
    
        p[k] = species_survival(core_, m , k , node_set) / len(node_set)

    plt.plot(p.keys(), p.values(), label = label)
    
plt.title(m)
plt.xlabel('k')
plt.ylabel('prob survival')
plt.legend()
plt.show()

## nodes in km core per category

k = 7
m = 4

df = EC.entities.loc[list(core[m][k][0])]

for typ in df['Category of registration'].unique():
    print(typ)
    print( df[df['Category of registration'] == typ].index, '\n'
    )
    
## Table rank centralities
centralities = ['Strength', 'Hypercoreness']
df_cent_rank = pd.DataFrame( columns = ['Name'] + centralities)
max_rank = 30
ranks = range(1,max_rank+1)
    
cent = centralities[1]
df = EC.entities.sort_values(cent, ascending = False)[['TR Name','Category of registration', 'NACE']].head(max_rank)
print(df)    

##

max_rank = 30
ranks = range(1,max_rank+1)
df = pd.DataFrame({'Nodes': EC.entities.sort_values(cent, ascending = False).head(max_rank).index , 'Sector': ['Energy' for _ in range(max_rank)]})
df.index = ranks
print(df.to_latex())  
## Membership of node
EC.H.nodes.memberships['ENGIE']
##
plt.close()
l = list(EC.entities['Hypercoreness'].dropna())
l.sort(reverse= True)
plt.plot(range(len(l)), l)
plt.show()
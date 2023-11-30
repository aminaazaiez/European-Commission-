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
meetings_path = path + 'data/meetings/'
EU = EuropeanCommission(meetings_path)


# Transparency Register
TR = TransparencyRegister()

EU.add_data_to_entities_from(TR.data, [ 'TR Country','Category of registration', 'Level of interest', 'Fields of interest', 'Lobbying cost' , 'Members FTE'] )
EU.entities['Category of registration'] = EU.entities['Category of registration'].str.replace('&' ,'and')

# Orbis
orbis = Orbis()

df = orbis.matched_names.merge(orbis.company_data, left_on = 'BvD ID', right_on = 'BvD ID', how = 'right')

EU.add_data_to_entities_from(df, right_on = 'Company name', columns = [ 'Orbis Country', 'Country ISO', 'City', 'BvD ID','BvD sectors', 'NAICS', 'Last year', 'Revenue', 'Shareholders funds',
       'Assets', 'Nb employees', 'corporate group size', 'Entity type',
       'GUO Name','GUO Country ISO', 'GUO Type', 'GUO NAICS', 'GUO Revenue', 'GUO Assets',
       'GUO Nb employees'])
       
       
#Centralities

centralities = pd.read_csv (path +'Programs/entities_centrality.csv')
columns  = ['Betweenness', 'Closeness', 'Degree','Strength']
EU.add_data_to_entities_from(centralities,columns)

##################### Save file for RStudio #################
df = EU.get_companies()[['Name','Betweenness', 'Closeness', 'Degree','Strength', 'TR Country','Members FTE', 'BvD sectors', 'Revenue', 'Assets', 'Nb employees', 'Level of interest', 'Fields of interest' ]].dropna().copy()


#Split catagorial data

levels = split_column(df, 'Level of interest', prefix = 'Level')
df.drop('Level of interest', axis = 1, inplace = True)
fields = split_column(df , 'Fields of interest', prefix = 'Field')
df.drop('Fields of interest', axis = 1, inplace = True)

df = pd.concat( [df , levels, fields], axis = 1)

#df = df [['Name','Betweenness', 'Closeness', 'Degree','Strength','Members FTE', 'Revenue', 'Assets', 'Nb employees']]
df.to_csv(path + 'Programs/data_test_R.csv', index = False)

fields.to_csv(path +'Programs/fields.csv', index = False)

###################### Computations #####################

B = Hnx_2_nxBipartite(EU.hypergraph)
nx.write_gexf(B, path+"/Programs/Bipartite.gexf")

## Centrality measures ##
centralities = [degree, strength, betweenness, closeness]
for algo in centralities:
    print(algo)
    cent = algo(EU.hypergraph)
    cent = cent /np.linalg.norm(cent)
    EU.entities = pd.concat([EU.entities, cent], axis = 1)
    

for mode in ['linear', 'log exp', 'max']:
    print(mode)
    cent = eigenvector(EU.hypergraph, mode)
    cent = cent /np.linalg.norm(cent)
    EU.entities = pd.concat([EU.entities, cent], axis = 1)
    
EU.entities = EU.entities.loc[list(EU.hypergraph.nodes())]    
    
## Save results
# hyperedges_carac.to_csv(path + 'Programs/hyperedge_carac.csv')
# EU.entities.to_csv(path + 'Programs/entities_centrality.csv')
# EU.entities.to_csv(path + 'Programs/entities_carac.csv', index = False)


#################### Vizualisation Results#####################
##
#plot degrees hist
feature = 'Strength'
bins = np.arange(EU.entities[feature].min(), EU.entities[feature].max(), 0.01)
kwargs = {'alpha' : 0.5, 'bins' : bins, 'edgecolor' : 'k'}
plt.close()
types = EU.entities['Type'].unique()
for t in types :
    EU.entities[ EU.entities['Type'] == t][feature].plot(kind ='hist', **kwargs, label =t)

plt.legend()
plt.xlabel('%s of entities'%feature)
plt.yscale('log')
plt.show()


## Hist plot of centralities
# plt.close(fig)
centralities = ['Closeness', 'Degree', 'Betweenness']
for centrality in centralities:
    entity_types = EU.entities['Type'].unique()
    fig, ax = plt.subplots(1,2, sharey = True, sharex = True , figsize = (7,3))
    for i , entity_type in enumerate( entity_types ):
        EU.entities[EU.entities['Type'] == entity_type][centrality].plot(kind ='hist', ax =ax[i], alpha = 0.5, bins= 20)
        ax[i].set_title(f"{entity_type}")
        ax[i].set_yscale('log')
    fig.suptitle(centrality) 
    
    plt.tight_layout()
    fig.savefig(path + 'Programs/Figures/'+centrality+'.pdf')
plt.show()

## Counts 

def count_and_percent(df , column ):
    df = df[column]
    return(pd.concat( (df.value_counts(dropna = False) , df.value_counts( dropna = False, normalize = True) ), axis = 1))

feature = 'Category of registration'
print( count_and_percent( EU.get_orga() , feature ))

## OLS regretion 
y ='Degree'
#x = np.sort(list(levels))
x = ['TR Country' ]
    
data  = EU.entities[EU.entities['Category of registration'] == 'Companies and groups' ].dropna(subset = x).dropna(subset = y)


Y = data[y]
#X = data['Nb employees']


X_cat = data[x]
X_cat = pd.get_dummies( X_cat , drop_first=True, dtype = int)

# # If two type of variable, contineous and categorial
#X = pd.concat([X, X_cat] , axis = 1)

X = sm.add_constant(X_cat)
results = sm.OLS(Y, X).fit()


print(results.summary().as_latex())
)

##### Mean groupby category
fields = split_column(EU.entities , 'Fields of interest')
y ='Strength'
x = fields
x_label = 'Fields of interest'

def _groupby(x , feature):
    return( pd.concat( [x , EU.entities] , axis =1).groupby(feature, dropna = False)[y])

mean_groupby = { feature : _groupby(x , feature).sum()[1] for feature in x}
std_groupby = { feature  : _groupby(x , feature).std()[1] for feature in x}
sem_groupby = { feature  : _groupby(x , feature).sem()[1] for feature in x}

print(pd.DataFrame({ x_label: mean_groupby.keys(), 'Mean': mean_groupby.values() ,  'Std' : std_groupby.values() , 'Sem' : sem_groupby.values()}).sort_values(by= 'Mean' , ascending=False))

df = EU.entities.groupby('Category of registration', dropna = False)[y]
#float_format="{:.2e}".format

## Exclusive categories
y = 'Strength'
x = 'TR Country'


df = EU.get_orga()[EU.get_orga()['Category of registration']== 'Companies and groups'].groupby(x)[y]

print(df.sum().sort_values(ascending = False).astype(int).head(50))


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

df = pd.concat([EU.entities , levels], axis = 1)
df[ (df ['TR Country'] == 'BELGIUM') & (df['European'] == 1 )].groupby('Category of registration')[y].sum()
## x vs y scatter per category
df = EU.get_orga()

x = 'Strength'
y = 'Betweenness'
plt.figure(figsize = (5,3))

business = ['Trade and business associations', 'Companies and groups' ]
category_colors = { category : 'blue'  if category in business else 'red' for category in df['Category of registration'].unique()}


# Create a scatter plot using the category_colors dictionary
for category, color in category_colors.items():
    category_data = df[df['Category of registration'] == category]
    plt.scatter(category_data[x], category_data[y], label=category, color=color, alpha = 0.1)
plt.ylabel('$t_i$')
plt.xlabel('$s_i$')
plt.legend(['Business interest', 'others'])
plt.tight_layout()
    
plt.show()



## Correlation btw hyperedge carac
hyperedges_carac  = pd.read_csv(path + 'Programs/hyperedge_carac.csv')
hyperedges_carac['Cardinality'].mean()
hyperedges_carac['Cardinality'].std()
hyperedges_carac[['Cardinality', 'Betweenness']].corr()


## Cumulative ditribution fonction of centralities according to category of registration
def cdf (ax, data, x_label , label, color):
    N=len(data)
    # sort the data in ascending order
    x = np.sort(data)
    # get the cdf values of y
    y = 1 - (np.arange(N) / float(N))
    # plotting
    ax.set_xlabel(x_label)
    ax.set_ylabel('CDF of %s' %x_label.lower())
    ax.plot(x, y, marker='o', markersize = 2, linewidth =0.4, color = color, label = label)

    #ax.title('cdf of %s in log log scale' %label)
    ax.set_yscale('log')
    ax.set_xscale('log')
    return(ax)
    
    
feature  = 'Closeness'
try:
     plt.close(fig)
except:
    None
fig, ax = plt.subplots( figsize = (5,5))
data = EU.entities[['Name', feature, 'Category of registration']].dropna()    
categories = data['Category of registration'].unique()
colors = sns.color_palette("hls", len(categories))
for category , color in zip (categories , colors):
    ax = cdf (ax, data[ data['Category of registration'] == category][feature] , feature, category, color)
    ax.legend()
fig.show()

## Correlation centralities and financial data
fin_data  =  ['Revenue', 'Nb employees', 'Assets']
net_data = ['Betweenness', 'Closeness', 'Degree']

for column in fin_data + net_data :
    EU.entities[column] = EU.entities[column].astype(float)

# global correlation
print(EU.entities[fin_data + net_data].corr( numeric_only = True).iloc[:len(fin_data), len(net_data):].to_latex())

#Correlation grouped by category of registration


idx = np.array([ i % (len(fin_data) + len(net_data)) in np.arange(len(fin_data)) for i in range (len(EU.entities['Category of registration'].dropna().unique()) * (len(net_data) + len(fin_data)))])

format_c1 = '{6cm}'
print(EU.entities.groupby('Category of registration')[fin_data + net_data].corr( numeric_only = True).iloc[idx,  len(net_data):])
#.to_latex(float_format="%.2f", longtable = True, column_format = 'p' + format_c1 + 'l' +'r' *len(net_data)).replace('{*}' , format_c1) )

## Correlation btw centralities and category of registration
from scipy import stats
levels = split_column(EU.entities , 'Level of interest')

net_data = ['Betweenness', 'Closeness', 'Degree']
for level in ['Global', 'European' , 'National', 'Regional/local']:
    print(level)
    print(EU.entities.dropna(subset = 'Level of interest')[net_data].corrwith(levels[level], method = stats.pointbiserialr))
    
    
##
df = EU.entities[EU.entities['Category of registration' ]== 'Companies and groups'].copy()
df_max = EU.entities[EU.entities['Category of registration' ]== 'Companies and groups'].copy()

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

df = EU.entities[EU.entities['Category of registration' ]== 'Companies and groups'].copy()
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

###

EU.entit
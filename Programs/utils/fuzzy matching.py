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
from classes.european_commission import *

path = '/home/azaiez/Documents/Cours/These/European Commission/'


## Load Data
EU = EuropeanCommission()
meetings_path = path + 'data/meetings/'

EU.initialize( meetings_path)

transparency_register = pd.read_excel(path + 'data/transparency_register.xls', engine='xlrd')
EU.add_data_to_ententies_from(transparency_register, ['Category of registration', 'Level of interest' , 'Head office country' ])
EU.entities.rename(columns = {'Head office country' : 'Country'}, inplace = True )

ciq = pd.read_csv(path + 'data/ciq_firms_clean.csv')
ciq = ciq.groupby(['Name', 'ciq_country']).agg({
    'revenues': 'sum',
    'nb_employees': 'sum'
}).reset_index()
ciq.to_csv(path + 'data/ciq_regrouped.csv')

## Clean Data

import re
from unidecode import unidecode
from cleanco import basename

organizations = EU.entities[ EU.entities['Type'] == 'Organization'].dropna().reindex()

def preprocess_name(name):
    # Remove leading/trailing whitespaces
    name = name.strip()

    # Convert to lowercase for case insensitivity
    name = name.lower()

    # Convert non unicode to unicode
    name =  unidecode(name)
    #unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode()


    # Remove suffixes
    if len(basename(name)) > 0 :

        name = basename(name)

    name = re.sub('international' ,'', name)
    # Remove special characters and extra spaces
    #name = re.sub(r'[^a-zA-Z0-9\s]', '', name)

    return name

organizations['cName'] = [preprocess_name(name) for name in organizations['Name']]
ciq['cName'] =  [preprocess_name(name) for name in ciq['Name']]


##Countries matching
dfCountries = fpd.fuzzy_merge(
                            pd.DataFrame({'EU_country' : organizations['Country'].dropna().unique()}),
                            pd.DataFrame({'ciq_country': ciq['ciq_country'].unique()}),
                            left_on = [ 'EU_country'],
                            right_on =['ciq_country'],
                            ignore_case = True,
                            keep_left = 'match' ,
                            join = 'full-outer'
                            )

#dfCountries.to_csv(path + 'data/matched_countries.csv' )


## fuzzy pandas matching for companies names
import fuzzy_pandas as fpd

thres = 0.8

dfResults = fpd.fuzzy_merge(organizations.dropna(), ciq,
                            left_on = ['cName'],
                            right_on =['cName'],
                            ignore_case = True,
                            keep_left = ['Name'] ,
                            keep_right = ['Name'],
                            method = 'exact' ,
                            ignore_order_words = True,
                            )

#Counter([ organizations [organizations[ 'Name' ]== name]['Category of registration'].values[0] for name in dfResults.iloc[:,0]] ) # number of matches per category of registration
dfResults.columns = ['trName' , 'ciqName' ]

dfResults.to_csv(path +'data/exact_match_organization.csv')



## Search non alphanumeric caraters in companies name


def contains_non_alphanumeric(text):
    return bool(pd.Series(text).str.contains('[^a-zA-Z0-9, -./]+').any())
companies[ companies['Name'].apply(contains_non_alphanumeric)]




## Translate


from googletrans import Translator
translator = Translator()
names = []
for i, name in enumerate(EU.entities['Name']):
    names.append(translator.translate(name).text)
    if  i % 100 ==0:
        print(i)


df = pd.DataFrame({'Original' : list(EU.entities['Name']) , 'Translated' : names})
#df.to_json(path+'data/EU_entities_eng.json', orient = "records")

## Match EU entities and transparency register
print( 'number of organizations appearing in the transparency register =' ,
len(EU.entities[EU.entities['Type'] == 'Organization'] )  - (len( set(EU.entities['Name']) - set(transparency_register[('Name')])))   ,
'\nNumber of organization in total =' ,
len(EU.entities[EU.entities['Type'] == 'Organization'] ))

## Math EU entities and ciq
# With original names
companies = EU.entities[EU.entities['Category of registration'] == 'Companies & groups']
print( 'Number of companies in ciq =' ,
         len( set(companies['Name']) & set(ciq['Name'])) ,
        '\n', 'Number of companies = ',
        len( set(companies['Name']))
)

# With translated names
trans_entities = EU.entities.copy()
trans_entities['Name'] = names
trans_companies = trans_entities[trans_entities['Category of registration'] == 'Companies & groups']

print( 'Number of companies in ciq =' ,
        len((set(trans_companies['Name']) | set(companies['Name']) )& set(ciq['Name'])) ,
        '\n', 'Number of companies = ',
        len( set(trans_companies['Name']))
)

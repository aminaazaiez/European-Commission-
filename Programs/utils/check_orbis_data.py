import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/home/azaiez/Documents/Cours/These/European Commission/'

sys.path.insert(0, path +'Programs')
from classes.european_commission import *
from classes.orbis import *
from classes.transparency_register import *

EU = EuropeanCommission()
meetings_path = path + 'data/meetings/'

EU.initialize( meetings_path)

n_orga = len(EU.entities[EU.entities['Type'] == 'Organization'])

TR = TransparencyRegister()
TR.initialize()

EU.add_data_to_ententies_from(TR.data, ['TR Country', 'Category of registration', 'Level of interest'])

## Load Orbis Data
orbis = Orbis()
orbis.initialize()


#Match EU and orbis data
df = orbis.matched_names.merge(orbis.company_data, left_on = 'BvD ID', right_on = 'BvD ID', how = 'right')

EU.add_data_to_ententies_from(df, right_on = 'Company name', columns = ['Orbis Country', 'BvD ID','BvD sectors', 'NAICS', 'Last year', 'Revenue', 'Shareholders funds',
       'Assets', 'Nb employees', 'corporate group size', 'Entity type',
       'GUO Name', 'GUO Country ISO', 'GUO Type', 'GUO NAICS', 'GUO Revenue', 'GUO Assets',
       'GUO Nb employees'])


## Duplicated BvD

IDS = EU.entities.dropna(subset ='BvD ID')[EU.entities.dropna(subset ='BvD ID').duplicated(subset = 'BvD ID')]['BvD ID']
mistyped = {}
manual_check = []
for ID in list(IDS):
    df = EU.entities[EU.entities['BvD ID'] == ID]
    replace = df[ df['TR Country'].isna()]['Name'].values
    keep =  df[df['TR Country'].notna()]['Name'].values
    if len( replace) == 1 and len(keep) == 1:
        mistyped[ replace[0]] = keep[0]
    elif len(replace) == 0:
        manual_check += list(keep)
        print(keep)
    elif len(keep) == 0:
        manual_check += list(replace)

print('mistyped = ', mistyped, '\n\n')
print('Manual check :', manual_check)

##
IDS = orbis.get_BvD_matched_companies()
with open (path + 'data/Orbis/BvD.txt', 'w') as f:
     f.write(IDS)

##
TR_orga = EU.entities.dropna( subset ='TR Country')
Orbis_orga = EU.entities.dropna(subset = 'Orbis Country')

def percent(n, total):
    return('( %.2f ' %(100*n/total) + '\%)')


print('Total number of organizations = ', n_orga , '\\\\')
print('Number of matched organizations in Orbis = ', len(Orbis_orga) , percent(len(Orbis_orga), n_orga), '\\\\')


print('Number of matched organizaiton in TR = ' , len(TR_orga), percent ( len(TR_orga) , n_orga), '\\\\')

print('Number of organization in TR but not in Orbis = ' , len( set(TR_orga['Name']) - set(Orbis_orga['Name'])) , '\\\\')

print('Number of organization matched in Orbis but not in TR = ' , len( set(Orbis_orga['Name']) - set(TR_orga['Name']) ) , '\\\\')

print('Number of organization with known revenue = ' , len(Orbis_orga.dropna(subset = 'Revenue')), percent (len(Orbis_orga.dropna(subset = 'Revenue')), n_orga), '\\\\')

print('Number of organization with known nb employees = ' , len(Orbis_orga.dropna(subset = 'Nb employees')), percent (len(Orbis_orga.dropna(subset = 'Nb employees')), n_orga), '\\\\')

print('Number of organization with known assets = ' , len(Orbis_orga.dropna(subset = 'Assets')), percent (len(Orbis_orga.dropna(subset = 'Assets')), n_orga), '\\\\')



print(pd.concat([EU.entities['Category of registration'].value_counts().rename('All'),
                    EU.entities.dropna(subset = 'Revenue')['Category of registration'].value_counts().rename('Known Revenu') ,
                    EU.entities.dropna(subset = 'Nb employees')['Category of registration'].value_counts().rename('Known Nb employees') ,
                    EU.entities.dropna(subset = 'Assets')['Category of registration'].value_counts().rename('Known Assets') ], axis = 1 ).to_latex( column_format = '{lp{1cm}p{1cm}p{1cm}p{1cm}}'))


## Correlation between number of employees, revenues and assets
print(orbis_data[['Revenue' , 'Nb employees', 'Assets']].corr().to_latex( float_format="%.2f"))

print(orbis_data.groupby('BvD sectors')[['Revenue' , 'Nb employees', 'Assets']].corr().to_latex( float_format="%.2f", longtable = True))


## Merge orbis data and matched company names
df = orbis_matched_names.merge(orbis_data, left_on = 'Matched company name', right_on = 'Name', how = 'right')
##
orbis_data.to_csv(path + 'data/orbis_data.csv')
orbis_matched_names.to_csv(path + 'data/matched_names.csv')
## Problem with duplicated entities in eu data
c = Counter(orbis_data['Name'])
for (orga , count) in c.items():
     if count > 1:
        print(orga)


        ##Check if there are matched names by orbis but not fin in orbis data (maybe du to reasher setups)
l = []
for element in set(orbis_matched_names['Matched company name'].dropna()) - set(orbis_data['Name'].dropna() ):
    print(element)


## Check type of TR corresponds to BVD sector
df = EU.entities.dropna(subset = 'Category of registration').dropna(subset = 'BvD sectors')


for  name, tr_cat, orbis_cat in zip(df['Name'] , df['Category of registration'], df['BvD sectors']):
    if (tr_cat != 'Companies & groups')  and (orbis_cat != 'Public Administration, Education, Health Social Services' and orbis_cat !='Business Services' ):
        print(name, tr_cat, orbis_cat)
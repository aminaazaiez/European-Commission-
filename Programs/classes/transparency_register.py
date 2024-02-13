import os
import pandas as pd
import numpy as np


# l = []
# for cost in TR.data['Lobbying cost']:
#     if isinstance(cost ,float):
#         l.append(cost)
#     elif '-'in cost:
#         print([int(c) if c != '' else np.nan for c in cost.split('-')])


def create_transparency_register(path):
    file_names = os.listdir(path+'Transparency register/' )
    file_names.sort() # Sort to keep the latest information about organization

    TR = pd.DataFrame()

    mapper_previous2new = {
    'Registration date:':'Registration date',
    'Subsection':'Category of registration',
    '(Organisation) name':'Name',
    'Legal status:':'Form of the entity',
    'Website address:':'Website URL',
    'Belgium office address':'EU office address',
    'Belgium office city':'EU office city',
    'Belgium office post code': 'EU office post code',
    'Belgium office post box':'EU office post box',
    'Belgium office phone':'EU office phone',
    'Goals / remit':'Goals',
    'EU initiatives':'EU legislative proposals/policies',
    'Relevant communication':'Communication activities',
    'High-level groups':'Expert Groups',
    'Inter groups':'Intergroups and unofficial groupings',
    'Number of persons involved:':'Members',
    'Full time equivalent (FTE)':'Members FTE',
    'Number of EP accredited persons':'Number of EP acredited Person',

    'Membership':'Is member of: List of associations, (con)federations, networks or other bodies of which the organisation is a member',

    'Member organisations':'Organisation Members: List of organisations, networks and associations that are the members and/or  affiliated with the organisation'
    }

    mapper_shorten= {'Head office country' : 'TR Country',
                        'Name': 'TR Name'}



    for file in file_names:

        try :
            # New version of the transparency register
            df = pd.read_excel(path +'Transparency register/' +file, engine='xlrd', index_col = 'Identification code' )
        except :

            # Previous version of the transparency register
            df = pd.read_excel(path +'Transparency register/' + file, engine='xlrd', index_col = 'Identification number:' )
            df.index = df.index.rename('Identification code')
            df.rename(columns = mapper_previous2new, inplace = True)

        TR = pd.concat([TR, df]).groupby('Identification code').last()
    TR.rename(columns = mapper_shorten, inplace = True)

    TR['TR Country'] = TR['TR Country'].str.upper()
    TR['Level of interest'] = TR['Level of interest'].str.title()
    return(TR)

def TransparencyRegister(path, columns = None,reload = False):
    if reload :
        TR = create_transparency_register(path)
        TR.to_csv(path + 'Transparency_register.csv')
    else :
        TR = pd.read_csv(path + 'Transparency_register.csv', index_col = 'Identification code')
    if columns is not None:
        TR = TR[columns]
    return(TR)

    # Save
# df = meetings['TR ID'].apply(lambda x : set(x) - set(TR.index))
# remaining_meetings = meetings.loc[df.mask( df ==set()).dropna().index]
# remaining_meetings.to_csv('/home/azaiez/Documents/Cours/These/European Commission/data' + 'remaining_meetings.csv')

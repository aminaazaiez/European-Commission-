import pandas as pd
import os
import itertools
import hypernetx as hnx
import networkx as nx
from collections import Counter
from collections.abc import Sequence, Iterable

date = '2019-12-01'
mistyped =  {'Disinformation Index Ltd': 'The Global Disinformation Index', 'Fiat Chrysler Automobiles': 'Stellantis', 'Confederatia Patronala Concordia': 'Confederația Patronală CONCORDIA', 'PostNL Holding B.V.': 'PostNL Holding B.V>', 'Air France KLM': 'Air France-KLM', 'Open Finance Association Europe': 'Open Finance Association', 'Estonian Renewable Energy Association': 'Portuguese Renewable Energy Association', 'NV Nederlandse Spoorwegen': 'Nederlandse Spoorwegen', 'Fair Trade Advocacy Office': 'Stichting Fair Trade Advocacy Office', 'Tweeddale Advisors': 'Tweeddale Advisors Ltd', 'Electrolux Home Products': 'Electrolux Home Products Europe', 'Stowarzyszenie Rzeźników i Wędliniarzy RP': 'Stowarzyszenie Rzeźników i Wędliniarzy Rzeczypospolitej Polskiej', 'ORGANIZACIÓN NACIONAL DE CIEGOS ESPAÑOLES': 'ORGANIZACION NACIONAL DE CIEGOS DE ESPAÑA', 'POLSKI ZWIĄZEK HODOWCÓW I PRODUCENTÓW BYDŁA MIĘSNEGO': 'Polski Związek Hodowców i Producentów Bydła Mięsnego', 'Internet Corporation for Assigned Names and Numbers': 'The Internet Corporation for Assigned Names and Numbers', 'Bundesverband Güterkraftverkehr Logistik und Entsorgung (BGL) e. V.': 'Bundesverband Güterkraftverkehr Logistik und Entsorgung (BGL) e.V.', 'Ragn-Sells': 'Ragn Sells AS', 'Eesti Tööandjate Keskliit': 'EESTI TÖÖANDJATE KESKLIIT MTÜ', 'Oxfam-Solidarité / Oxfam-Solidariteit': 'Oxfam-en-Belgique / Oxfam-in-België', 'Atlantic Council of the United States, Inc.': 'Atlantic Council of the United States, Inc', 'Cboe Europe Limited': 'CboeEurope', 'Seas At Risk vzw': 'Seas At Risk', 'Institutional Limited Partners Association (ILPA)': 'Institutional Limited Partners Association'}

manual_check = {'Compassion in World Farming Brussels' : 'Compassion in World Farming International' ,
                'ASOCIACION AGRARIA JOVENES AGRICULTORES MALAGA' :'Asociación Agraria Jóvenes Agricultores',
                'ClearVAT': 'eClear AG',
                'Yleisradio Oy' : 'Nordvision'}


def search_and_replace_tuple(tuple_value, replacement_dict):
    new_tuple = tuple(replacement_dict.get(element, element) for element in tuple_value)
    return new_tuple

def mask_by_str(df : pd.DataFrame, column :str, value : str):
    return(df.dropna(subset = column)[EU.entities.dropna(subset = column)[column].str.contains(value)])

def search_str_in_column_of_tuple(dataframe : pd.DataFrame, column_name : str, search_string : str):
    filter_condition = dataframe[column_name].apply(lambda x: any(search_string in element for element in x))
    filtered_df = dataframe[filter_condition]
    return filtered_df


def merge_iterable_of_tuples( l1 : Iterable , l2 : Iterable):
    return( [ tuple_1 + tuple_2 for tuple_1 , tuple_2 in zip ( l1 , l2) ] )


def hyperedges2biartite( edges , edge_weights):
    edge_name, node_name, weight = [] ,[], []

    for i , (edge, edge_weight)  in enumerate( zip (edges , edge_weights)):
        for node in edge:
            edge_name.append(i)
            node_name.append(node)
            weight.append(edge_weight)
    return(pd.DataFrame( {'edge' : edge_name , 'node' : node_name, 'weight' : weight}))




def split_column(df : pd.DataFrame, column : str, sep :str =', ' ,  prefix : str = None):
    items = set(itertools.chain.from_iterable([item.split(sep) for item in df[column].dropna().unique()]))
    df = pd.DataFrame( df[column])

    for item in items:
        if prefix :
            col_name = prefix + '_' + item
        else :
            col_name = item
        # Initialize new columns with 0
        df[col_name] = pd.Series()
        df[col_name] = df[col_name].where(df[column].isna(), 0)
        # Use .str.contains() to set the values to 1 where applicable
        df.loc[ df[column].str.contains(item, case=False , regex = True) == True, col_name] = 1 # to deal with nan

    df.drop( column , axis = 1 , inplace = True)
    return(df)

def toy():
    edges = [('A', 'B', 'C'), ('C', 'D', 'E', 'F'), ('B','E','F', 'G')]
    weights = [3, 1,10]

    df_bipartite = hyperedges2biartite(edges, weights)

    hypergraph = hnx.Hypergraph( df_bipartite , edge_col = 'edge' , node_col = "node", cell_weight_col="weight",  sort=False)

    # Add weights and strengths as attributes to edges and nodes
    df =hypergraph.dataframe.groupby('edge')['weight'].first()
    hypergraph.edge_props.update(df)
    return(hypergraph)


class EuropeanCommission:
    def __init__(
        self,
        meetings_path : str ,
        meetings : pd.DataFrame = pd.DataFrame(),
        entities : pd.DataFrame = pd.DataFrame(),
        hypergraph : hnx.Hypergraph = hnx.Hypergraph(),
        ):
        self.meetings = meetings
        self.entities = entities
        self.hypergraph = hypergraph

        #Add commissioner and cabinet member schedule
        for file in [meetings_path + 'commissioners.jsonl' ,  meetings_path + 'cabinet_members.jsonl', meetings_path + 'directorate_general.jsonl']:
            data= pd.read_json( file, lines = True)
            data = data[['EU Member', 'Date', 'Entities', 'Subjects']]
            self._add_meetings(data)

        #Add entities
        eu_members = set(itertools.chain.from_iterable( self.meetings['EU Member'] ))
        organizations = set(itertools.chain.from_iterable( self.meetings['Entities'] ))

        self.entities = pd.concat([self.entities ,  pd.DataFrame( { 'Type' : ['EU Member' for _ in eu_members ] } , index = list(eu_members  )) ])
        self.entities = pd.concat([self.entities ,  pd.DataFrame( {'Type' : ['Organization' for _ in organizations] } , index = list(organizations))  ] )
        self.entities.index.rename('Name', inplace = True)

        #Generate hypergraph
        self.generate_hypergraph()



    def _add_meetings(self,  data : pd.DataFrame):
        data['EU Member'] = [tuple( data['EU Member'][i]) for i in range (len(data))]
        data['Entities'] = [tuple( data['Entities'][i]) for i in range (len(data))]
        data['Subjects'] = [tuple( data['Subjects'][i]) for i in range (len(data))]
        self.meetings = pd.concat([self.meetings , data], ignore_index= True)
        # search and replace specific organization that misstyped
        self.meetings['Entities'] = self.meetings['Entities'].apply(lambda x : search_and_replace_tuple(x, mistyped))
        self.meetings['Entities'] = self.meetings['Entities'].apply(lambda x : search_and_replace_tuple(x, manual_check))

        # Delete data previous to the actual commission
        self.meetings = self.meetings[self.meetings['Date']> date]



    def generate_hypergraph(self):
        hyperedges = merge_iterable_of_tuples ( self.meetings['EU Member'] ,  self.meetings['Entities'])
        #agregate by sum
        hyperedges = Counter(hyperedges)
        df_bipartite = hyperedges2biartite(hyperedges.keys() , hyperedges.values())

        self.hypergraph = hnx.Hypergraph( df_bipartite , edge_col = 'edge' , node_col = "node", cell_weight_col="weight",  sort=False)

        # Select the maximal connected subgraph
        components = list( self.hypergraph.connected_components())
        giant_component = max(components, key=len)
        self.hypergraph = self.hypergraph.restrict_to_nodes(giant_component)
        self.entities = self.entities.loc[list(giant_component)]

        # Add weights and strengths as attributes to edges and nodes
        df =self.hypergraph.dataframe.groupby('edge')['weight'].first()
        self.hypergraph.edge_props.update(df)


        # df = self.hypergraph.dataframe.groupby('node')['weight'].sum().rename('weight')
        # for i, node in enumerate (self.hypergraph.nodes ):
        #     self.hypergraph.nodes[node].strength = df[node]


    def add_data_to_entities_from(self, df : pd.DataFrame  , columns : Iterable[str] | str, right_on : str = 'Name', suffixes = (None, 'y')):
        df.index = df[right_on]

        # Add the values from df2 to df1
        if isinstance(columns , str):
            self.entities[columns] = df[columns]

        elif isinstance(columns , Iterable):
            for column in columns :
                self.entities[column] =  df[column ]



    def save_orga_names_batchs(self, columns : list, path_file : str):
        df = self.entities[EU.entities['Type'] == 'Organization'][colmuns]
        rows_per_file = 1000

        # Calculate the total number of files needed
        total_files = len(df) // rows_per_file + (len(df) % rows_per_file > 0)

        # Split the DataFrame into smaller chunks and save each chunk as a separate file
        for i in range(total_files):
            start_idx = i * rows_per_file
            end_idx = start_idx + rows_per_file
            chunk_df = df.iloc[start_idx:end_idx]
            chunk_df.to_csv(path_file + f'organization_names_{i + 1}.csv', index=False, sep = '\t')

    def get_orga(self):
        return(self.entities[self.entities['Type'] == 'Organization'])

    def get_companies(self):
        return(self.entities[self.entities['Category of registration'] == 'Companies and groups'])


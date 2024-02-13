import pandas as pd
import os
import itertools
import hypernetx as hnx
import networkx as nx
from collections import Counter
from collections.abc import Sequence, Iterable
import xgi

date = '2019-12-01'

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


def hyperedges2biartite( edges_id ,  edges ):
    edge_name, node_name = [] ,[]

    for i , edge  in zip( edges_id , edges ):
        for node in edge:
            edge_name.append(i)
            node_name.append(node)
    return(pd.DataFrame( {'edge' : edge_name , 'node' : node_name}))


def split_column(df : pd.DataFrame, column : str, sep :str =', ' ,  prefix : str = None):
    items = set(itertools.chain.from_iterable([item.split(sep) for item in df[column].dropna().unique()]))
    df = pd.DataFrame( df[column])

    for item in items:
        if prefix :
            col_name = prefix + ' ' + item
        else :
            col_name = item
        # Initialize new columns with 0
        df[col_name] = pd.Series()
        df[col_name] = df[col_name].where(df[column].isna(), 0)
        # Use .str.contains() to set the values to 1 where applicable
        df.loc[ df[column].str.contains(item, case=False , regex = True) == True, col_name] = 1 # to deal with nan


    df.drop( column , axis = 1 , inplace = True)

    #df = df.astype('int')
    return(df)


def toy():
    edges = [('A', 'B', 'C'),('A', 'B', 'C'), ('C', 'D', 'E', 'F'), ('B','E','F', 'G')]
    H = xgi.Hypergraph( edges )

    return(H)


class EuropeanCommission:
    def __init__(
        self,
        data_path : str ,
        meetings : pd.DataFrame = pd.DataFrame(),
        entities : pd.DataFrame = pd.DataFrame(),
        H : hnx.Hypergraph = hnx.Hypergraph(),
        ):
        self.data_path = data_path
        self.meetings = meetings
        self.entities = entities
        self.H = H


        # Load EC represenatives meetings
        self.load_meetings()

        self.meetings.rename(columns = {
                'Name of EC representative' : 'EC member',
                'Date of meeting' : 'Date',
                'Transparency register ID' : 'TR ID',
                'Name of DG - full name' : 'Name of DG'}, inplace = True)

        self.meetings['EC member'] = [tuple( EC.split(',') ) for EC in self.meetings['EC member']]
        self.meetings['Title of EC representative'] = [tuple( EC.split(',') ) for EC in self.meetings['Title of EC representative']]

        self.meetings['TR ID'] = [tuple( EC.split(',') ) for EC in self.meetings['TR ID']]

        self.add_relative_commission()

        # Create entities dataframe

        self.create_entites_df()

        # Generate hypergraph
        self.generate_hypergraph()



    def load_meetings(self):

        # Load cabinet meetings

        file_name = 'Meetings of Commission representatives of the Von der Leyen Commission (2019-2024).xlsx'
        columns = ['Name of cabinet', 'Name of EC representative', 'Title of EC representative', 'Transparency register ID', 'Name of interest representative', 'Date of meeting']
        self.meetings = pd.read_excel(self.data_path + 'meetings/' + file_name, skiprows=[0], usecols=columns )


        # Load DG meetings

        file_name = 'Meetings of Directors-General of the European Commission.xlsx'
        columns = ['Name of DG - full name', 'Name of EC representative', 'Title of EC representative', 'Transparency register ID', 'Name of interest representative','Date of meeting']
        self.meetings = pd.concat([self.meetings , pd.read_excel(self.data_path + 'meetings/' + file_name, skiprows=[0], usecols=columns ) ] ,  ignore_index= True)

        to_drop = ['Recovery and Resilience Task force', 'Regulatory Scrutiny Board' , 'Informatics', 'European Personnel Selection Office', 'Task Force for Relations with the United Kingdom']
        idx = self.meetings.loc[self.meetings['Name of DG - full name'].isin(to_drop)].index
        self.meetings.drop(index = idx, inplace  = True)

        # Filter by date

        self.meetings = self.meetings[self.meetings['Date of meeting'] > date]


    def add_relative_commission(self):
        # Add relative commission to cabinet
        self.meetings.loc[self.meetings['Name of DG'].isna(), 'Relative commission'] = self.meetings['Name of cabinet'].str.extract( r'Cabinet of (Commissioner|Vice-President|High Representative|Executive Vice-President|President) ([^,]+)', expand=False)[1]

        # Add relative commission to DG

        data = pd.read_csv(self.data_path + 'DGs_relative_com.csv')
        com_relative_DG = dict(zip(data['Name of DG'], data['Commissioner']))
        self.meetings.loc[self.meetings['Name of cabinet'].isna(), 'Relative commission'] = self.meetings['Name of DG'].map(com_relative_DG)



    def create_entites_df(self):
        self.entities = pd.DataFrame()

        ec_members = set(itertools.chain.from_iterable( self.meetings['EC member'] ))
        organizations = set(itertools.chain.from_iterable( self.meetings['TR ID'] ))

        self.entities = pd.concat([self.entities ,  pd.DataFrame( {'Type' : ['Organization' for _ in organizations] } , index = list(organizations))  ] )
        self.entities = pd.concat([self.entities ,  pd.DataFrame( { 'Type' : [ 'EC member' for _ in ec_members ] } , index = list(ec_members  )) ])
        self.entities.index.rename('Name', inplace = True)




    def generate_hypergraph(self):
        hyperedges = merge_iterable_of_tuples ( self.meetings['EC member'] ,  self.meetings['TR ID'])
        #agregate by sum
        hyperedges = {id_e : e for id_e , e in zip(self.meetings.index , hyperedges) }

        # Select the maximal connected subgraph

        self.H = xgi.Hypergraph( hyperedges )
        self.H.remove_nodes_from(self.H.nodes - xgi.largest_connected_component(self.H))

        self.entities = self.entities.loc[list(self.H.nodes)]
        self.meetings = self.meetings.loc[list(self.H.edges)]



    def collapse_entities(self, mapper):
        ''' mapper : dict of nodes to collapse'''
        def replace_element_in_tuple(x,mapper):
            return( tuple(mapper.get(item, item) for item in x))


        self.meetings['TR ID'] = self.meetings['TR ID'].apply(lambda x : replace_element_in_tuple(x,mapper))
        self.generate_hypergraph()




    def add_data_to_entities_from(self, data : pd.DataFrame  , columns : Iterable[str] | str, right_on : str = 'Name', suffixes = (None, 'y')):
        df  = data.copy()
       # df.index = df[right_on]
        idx = list( set(df.index ) & set (self.entities.index))

        # Add the values from df2 to df1
        if isinstance(columns , str):
            self.entities.loc[idx, columns] = df[columns]

        elif isinstance(columns , Iterable):
            for column in columns :
                self.entities.loc[ idx, column] =  df[column ]



    def EC_members_info(self , reload = False):
        ''' Returns a dataframe with columns Name, Title, Relative Commission for EC members'''
        if reload :
            df = pd.concat([pd.DataFrame({'Name': ec_member, 'Title': title, 'Commission': com}, index=[0]) for ec_members, titles, com in zip(self.meetings['EC member'],
        self.meetings['Title of EC representative'], self.meetings['Relative commission'])
                    for ec_member, title in zip(ec_members, titles)], ignore_index=True)
            df.drop_duplicates(inplace = True)
            df.to_csv(self.data_path + 'EC_members_info.csv', index = False )

        else :
            df = pd.read_csv(self.data_path + 'EC_members_info.csv')
        return(df)




    def get_orga(self):
        return(self.entities[self.entities['Type'] == 'Organization'])

    def get_companies(self):
        return(self.entities[self.entities['Category of registration'] == 'Companies and groups'])


    def save_orga_names_batchs(self, columns : list, path_file : str):
        df = self.get_companies()[columns]
        df.index.rename('TR ID' , inplace = True)
        rows_per_file = 100

        # Calculate the total number of files needed
        total_files = len(df) // rows_per_file + (len(df) % rows_per_file > 0)

        # Split the DataFrame into smaller chunks and save each chunk as a separate file
        for i in range(total_files):
            start_idx = i * rows_per_file
            end_idx = start_idx + rows_per_file
            chunk_df = df.iloc[start_idx:end_idx]
            chunk_df.to_csv(path_file + f'company_names_{i + 1}.csv',  sep = '\t', index = True)
import pandas as pd
import os
import itertools
import hypernetx as hnx
from collections import Counter
from collections.abc import Sequence, Iterable

def str2tuple(l : Iterable, sep = ', ' ):
    return( [ tuple(s.split(sep)) for s in l ] )



def merge_iterable_of_tuples( l1 : Iterable , l2 : Iterable):
    return( [ tuple_1 + tuple_2 for tuple_1 , tuple_2 in zip ( l1 , l2) ] )


def hyperedges2biartite( edges , edge_weights):
    edge_name, node_name, node_weight = [] ,[], []

    for i , (edge, edge_weight)  in enumerate( zip (edges , edge_weights)):
        for node in edge:
            edge_name.append(i)
            node_name.append(node)
            node_weight.append(edge_weight)
    return(pd.DataFrame( {'edges' : edge_name , 'nodes' : node_name, 'weights' : node_weight}))


class EuropeanCommission:
    def __init__(
        self,
        meetings : pd.DataFrame = pd.DataFrame(),
        entities : pd.DataFrame = pd.DataFrame(),
        hypergraph : hnx.Hypergraph = hnx.Hypergraph(),
        transparency_register = pd.DataFrame(),
        ciq = pd.DataFrame(),
        ):
        self.meetings = meetings
        self.entities = entities
        self.hypergraph = hypergraph
        self.transparency_register = transparency_register
        self.ciq = ciq

    def _get_eu_member_name(self,file_name : str):
        for i, world in enumerate(file_name.split(' ')):
            if 'President' in world :
                start = i+1
            elif 'Commissioner' in world :
                start = i+1
            elif 'Representative' in world :
                start = i+1
            if world == 'until':
                stop = i
        Commissioner = " ".join(file_name.split(' ')[start:stop])
        return(Commissioner)

    def _add_commissioner_meetings(self,  Commissioner : str , data : pd.DataFrame):
        com = pd.DataFrame({'Name' : [ tuple([Commissioner]) for i in range(len(data))]})
        data['Entities'] = str2tuple(data['Entities'] )

        meetings_ = pd.concat([com , data ], axis =1)
        self.meetings = pd.concat([self.meetings , meetings_], ignore_index= True)



    def _add_cabinet_members_meetings(self, data : pd.DataFrame):
        data['Name'] = str2tuple(data['Name'] )
        data['Entities'] = str2tuple(data['Entities'] )
        self.meetings= pd.concat([self.meetings , data], ignore_index= True)


    def _add_commissioner2entities(self , Commissioner : str):
        self.entities = pd.concat([self.entities , pd.DataFrame({'Name' : [Commissioner] , 'Type' : ['Commissioner']})] , ignore_index= True )


    def _add_cabinet_member2entities(self, data : pd.DataFrame):
        cabinet_members = set(itertools.chain.from_iterable(  data['Name'] ))
        self.entities = pd.concat([ self.entities , pd.DataFrame( {'Name':  list(cabinet_members ), 'Type' : ['Cabinet Member' for _ in range(len(cabinet_members))] })  ] , ignore_index = True)


    def _add_organizations2entities(self , data : pd.DataFrame):
        organizations = set(itertools.chain.from_iterable( data['Entities'] )) - set (self.entities['Name'])
        self.entities = pd.concat([self.entities ,  pd.DataFrame( {'Name':  list(organizations ), 'Type' : ['Organization' for _ in range(len(organizations))] })  ] , ignore_index = True)




    def add_commissioner_schedule(self, data : pd.DataFrame, file_name: str):

        Commissioner = self._get_eu_member_name(file_name)
        self._add_commissioner_meetings( Commissioner, data)
        self._add_commissioner2entities(Commissioner)
        self._add_organizations2entities(data)


    def add_cabinet_member_schedule( self, data : pd.DataFrame ):
        self._add_cabinet_members_meetings( data)
        self._add_cabinet_member2entities(data)
        self._add_organizations2entities(data)


    def remove_entity(self, entity : str ):
        #remove entity from self.entities
        self.entities = self.entities[ entities['Name'] != entity]
        #remove entity from self.meetings


    def generate_hypergraph(self):
        hyperedges = merge_iterable_of_tuples ( self.meetings['Name'] ,  self.meetings['Entities'])
        #agregate by sum
        hyperedges = Counter(hyperedges)
        df_bipartite = hyperedges2biartite(hyperedges.keys() , hyperedges.values())

        self.hypergraph = hnx.Hypergraph( df_bipartite , edge_col = 'edges' , node_col = "nodes", cell_weight_col="weights")


    def add_data_to_ententies_from(self, df : pd.DataFrame  , columns : Iterable[str] | str , key_column : str = 'Name'):

        # Merge df1 and df2 on the key column
        merged_df = self.entities.merge(df, on=key_column, how='left')

        # Add the values from df2 to df1
        if isinstance(columns , str):
            self.entities[columns] = merged_df[columns]

        elif isinstance(columns , Iterable):
            for column in columns :
                self.entities[column] = merged_df[column]




    def initialize(self, meetings_path : str ):
        #Add commissioner schedule
        meeting_com_path = meetings_path + '/Commissioners/'
        dirList = os.listdir(meeting_com_path)
        for file_name in dirList:
            data_file = pd.read_excel( meeting_com_path + file_name, engine='openpyxl', skiprows =1, usecols = ['Date of meeting', 'Entity/ies met', 'Subject(s)'])
            data_file = data_file.rename(columns = { 'Date of meeting' : 'Date', 'Entity/ies met' : 'Entities', 'Subject(s)' : 'Subjects'})
            self.add_commissioner_schedule( data_file,file_name)

        # #Add member of cabinet schedule
        # meetings_path_cabinet = meetings_path + '/Cabinets/'
        # dirList = os.listdir(meetings_path_cabinet)
        # for file_name in dirList:
        #     data_file = pd.read_excel( meetings_path_cabinet + file_name, engine='openpyxl', skiprows =1, usecols = ['Name', 'Date of meeting', 'Entity/ies met', 'Subject(s)'])
        #     data_file = data_file.rename(columns = { 'Name' : 'Name' , 'Date of meeting' : 'Date', 'Entity/ies met' : 'Entities', 'Subject(s)' : 'Subjects'})
        #     self.add_cabinet_member_schedule( data_file)

        #self.generate_hypergraph()



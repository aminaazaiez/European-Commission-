import pandas as pd
import os
import itertools


def get_commissioner_name(file_name):
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
     
     
def load_commissioner_schedules(path, meetings, entities):
    #Get data from Commissioner schedule
    dirList = os.listdir(path)
    for file in dirList:
        # get commisionner name
        Commissioner = get_commissioner_name(file)
        # add commissioner to entities
        entities = pd.concat([entities , pd.DataFrame({'Name' : [Commissioner] , 'Type' : ['Commissioner']})] , ignore_index= True )
        #get meeting data
        data_file = pd.read_excel( path + file, engine='openpyxl', skiprows =1, usecols = ['Date of meeting', 'Entity/ies met', 'Subject(s)'])
        
        com = pd.DataFrame({'Name' : [Commissioner for i in range(len(data_file))]})
        meetings_ = pd.concat([com , data_file ], axis =1)
        meetings = pd.concat([meetings , meetings_], ignore_index= True)
        
    return(meetings , entities)
    
    
def load_cabinet_member_schedules(path, meetings, entities):

    #Get data from Commissioner cabinet schedule
    dirList = os.listdir(path)
    for file in dirList:
        data_file = pd.read_excel( path+ file, engine='openpyxl', skiprows =1, usecols = ['Name','Date of meeting', 'Entity/ies met', 'Subject(s)'])
        meetings = pd.concat([meetings , data_file], ignore_index= True)
        cabinet_members = set(itertools.chain.from_iterable( [ members.split(', ') for members in  data_file['Name']] ))
        entities = pd.concat([ entities , pd.DataFrame( {'Name':  list(cabinet_members ), 'Type' : ['Cabinet Member' for _ in range(len(cabinet_members))] })  ] , ignore_index = True)
    return(meetings, entities) 
    
def add_organizations(meetings, entities):
    organizations = set(itertools.chain.from_iterable( [ members.split(', ') for members in  meetings['Entity/ies met']] ))
    entities = pd.concat([ entities , pd.DataFrame( {'Name':  list(organizations ), 'Type' : ['Organisation' for _ in range(len(organizations))] })  ] , ignore_index = True)
    return(entities)
    
def get_hyperedges(meetings):
    return( [ tuple( members.split(', ') + s.split(', ') ) for s , members in zip( meetings['Entity/ies met'] , meetings['Name']) ]) 

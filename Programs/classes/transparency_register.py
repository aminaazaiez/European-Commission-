import os
import pandas as pd
import numpy as np
from classes.european_commission import *



class TransparencyRegister:
    def __init__(

        self,
        path = '/home/azaiez/Documents/Cours/These/European Commission/data/'
        ):

        self.data  = pd.read_excel(path + 'transparency_register.xls', engine='xlrd').drop_duplicates(subset = 'Name')
        self.data.rename(columns = {'Head office country' : 'TR Country' , 'Head office city' : 'City', 'Annual costs for registers activity or total budget' : 'Lobbying cost'}, inplace = True )


# l = []
# for cost in TR.data['Lobbying cost']:
#     if isinstance(cost ,float):
#         l.append(cost)
#     elif '-'in cost:
#         print([int(c) if c != '' else np.nan for c in cost.split('-')])
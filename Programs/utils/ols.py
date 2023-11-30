import pandas as pd
import statsmodels.api as sm
from scipy.stats import yeojohnson, skew
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
path = '/home/azaiez/Documents/Cours/These/European Commission/'




def get_skewed_columns(df):
    """
    :param df: dataframe where the skewed columns need to determined
    :return: skew_cols: dataframe with the skewed columns
    """
    skew_limit = 0.5  # define a limit above which we will log transform
    skew_vals = df.skew()
    # Showing the skewed columns
    skew_cols = (skew_vals
                 .sort_values(ascending=False)
                 .to_frame()
                 .rename(columns={0: 'Skew'})
                 .query('abs(Skew) > {}'.format(skew_limit)))
    return skew_cols


def one_hot_encod(df : pd.DataFrame , col : list , to_drop = None):
    """
    :param df: input data in the form of a dataframe
    :param columns: list of columns to encode
    :return: df: dataframe with the encoded columns
    """
    if to_drop is not None:
        dummies  =  pd.get_dummies(df[col],  prefix_sep = '', prefix = '', dtype = int)
        dummies.drop(columns = to_drop, inplace = True)
    else :
        dummies  =  pd.get_dummies(df[col], drop_first=True, prefix_sep = '', prefix = '', dtype = int)

    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    return (df , list(dummies.columns))



def remove_outliers(df, col :str, threshold):
    df_ = df[col].value_counts(normalize = True).copy()*100

    cat_to_remove = list( df_[df_<threshold].index)
    com_to_remove = list(df[df[col].isin(cat_to_remove)].index)
    #print('removed category: ' , cat_to_remove)
    #print('number of outliers: ', len(com_to_remove))

    return(df[~df[col].isin(cat_to_remove)])


def transform_skewed(df, columns : list, method : str = 'log'):
    for col in columns:
        if method == 'yeo':
            df[col] = yeojohnson(df[col])[0]
        elif method == 'log':

            df = df[df[col] + 1>0]
            df[col] = np.log(df[col]+1)
    return(df)


def heatmap_corr (df , columns, **kwargs ):
    try:
        plt.close()
    except:
        pass
    sns.heatmap(df[columns].corr( ) ,  cmap = 'inferno_r' , vmin =0, vmax= 1 , **kwargs)
    plt.tight_layout()
    plt.show()

def xy_heatmap_corr (df , col1, col2, **kwargs ):
    try:
        plt.close()
    except:
        pass
    sns.heatmap(df[col1+col2].corr( ).loc[col1][col2] ,  cmap = 'inferno_r' , vmin =0, vmax= 1 , **kwargs)
    plt.tight_layout()
    plt.show()



data  = pd.read_csv(path + 'Programs/data_test_R.csv')

dependent = ['Strength', 'Degree', 'Betweenness', 'Closeness']

lobbying = ['Members FTE']
financials = ['Revenue', 'Nb employees', 'Assets']

country = ['TR Country']

sector = ['BvD sectors']

level = ['Level_Global', 'Level_Regional/local', 'Level_National',
       'Level_European']
field = ['Field_Economy', 'Field_Enlargement',
       'Field_Food safety', 'Field_Trans-European Networks',
       'Field_Business and industry',
       'Field_Agriculture and rural development', 'Field_Youth',
       'Field_Migration and asylum', 'Field_Climate action', 'Field_Trade',
       'Field_Sport', 'Field_Borders and security', 'Field_Culture and media',
       'Field_Humanitarian aid and civil protection', 'Field_Single market',
       'Field_finance and the euro', 'Field_Education and training',
       'Field_Fraud prevention', 'Field_Transport', 'Field_Taxation',
       'Field_Institutional affairs',
       'Field_International co-operation and development', 'Field_Customs',
       'Field_External relations', 'Field_Banking and financial services',
       'Field_Maritime affairs and fisheries', 'Field_Competition',
       'Field_Energy', 'Field_Public health', 'Field_Environment',
       'Field_Justice and fundamental rights', 'Field_Culture',
       'Field_Consumers', 'Field_Regional policy',
       'Field_Research and innovation', 'Field_Digital economy and society',
       'Field_Communication', 'Field_European neighbourhood policy',
       'Field_Budget', 'Field_Employment and social affairs',
       'Field_Foreign affairs and security policy']


country_groups = {
    'North America': ['CANADA', 'UNITED STATES'],
    'North Europe': ['NORWAY', 'FINLAND', 'DENMARK', 'SWEDEN',  'LITHUANIA', 'LATVIA', 'ESTONIA', 'UNITED KINGDOM', 'IRELAND'],
    'South Europe': ['ITALY', 'SPAIN', 'PORTUGAL', 'GREECE', 'CROATIA', 'CYPRUS'],
    'East Europe': ['CZECH REPUBLIC', 'POLAND',   'ROMANIA', 'HUNGARY', 'BULGARIA', 'SLOVAKIA', 'KOSOVO'],
    'West Europe': ['AUSTRIA', 'GERMANY', 'SWITZERLAND', 'BELGIUM', 'LUXEMBOURG', 'FRANCE', 'NETHERLANDS'],
    'East Asia': ['JAPAN', 'CHINA', 'HONG KONG'],
    'Other': ['SOUTH AFRICA', 'SINGAPORE']
}
country_aff = {}
for region in country_groups.keys():
    for c in country_groups[region]:
        country_aff [c] = region

#data['TR Country'] = data['TR Country'].apply(lambda x : country_aff[x])

threshold = 1 #percent
data = remove_outliers(data, 'TR Country', threshold)
data = remove_outliers(data, 'BvD sectors', threshold)

binary = field + level
continuous = lobbying+ financials
nominal = country + sector

X = data[continuous + binary + nominal]
X, sector= one_hot_encod(X, sector)
X, country= one_hot_encod(X, country)

Y = data[dependent]

# heatmap_corr (X ,  continuous , annot = True)
#xy_heatmap_corr(X, field, sector)


columns = list(get_skewed_columns(X[continuous]).index)
X = transform_skewed(X, columns,'log')
#heatmap_corr (X ,  continuous , annot = True)




X = X[field]
X = sm.add_constant(X)

Y = data[['Strength']]
Y = Y.loc[X.index]
results = sm.OLS(Y, X).fit()


print(results.summary())


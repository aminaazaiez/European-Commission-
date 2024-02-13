import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing
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

def get_similar_value_cols(df, percent=90):
    """
    :param df: input data in the form of a dataframe
    :param percent: integer value for the threshold for finding similar values in columns
    :return: sim_val_cols: list of columns where a singular value occurs more than the threshold
    """
    count = 0
    sim_val_cols = []
    for col in df.columns:
        percent_vals = (df[col].value_counts()/len(df)*100).values
        # filter columns where more than 90% values are same and leave out binary encoded columns
        if percent_vals[0] > percent :
            sim_val_cols.append(col)
            count += 1
    print("Total columns with majority singular value shares: ", count)
    return sim_val_cols


def one_hot_encod(df : pd.DataFrame , col : list , to_drop = None):
    """
    :param df: input data in the form of a dataframe
    :param columns: list of columns to encode
    :return: df: dataframe with the encoded columns
    """

    if to_drop == 'first':
        dummies  =  pd.get_dummies(df[col], drop_first=True, prefix_sep = '', prefix = '', dtype = int)
    elif to_drop is not None:
        dummies  =  pd.get_dummies(df[col],  prefix_sep = '', prefix = '', dtype = int)
        dummies.drop(columns = to_drop, inplace = True)

    else :
        dummies  =  pd.get_dummies(df[col], prefix_sep = '', prefix = '', dtype = int)

    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    return (df , list(dummies.columns))



def remove_outliers(df, col :str, threshold):
    df_ = df[col].value_counts(normalize = True).copy()*100

    cat_to_remove = list( df_[df_<threshold].index)
    com_to_remove = list(df[df[col].isin(cat_to_remove)].index)
    print('removed category: ' , cat_to_remove)
    print('number of outliers: ', len(com_to_remove))

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


data  = EC.get_companies()[[ 'Degree', 'Strength', 'Betweenness','EV_linear', 'EV_log exp', 'EV_max', 'Hypercoreness', 'TR Country','Members FTE',  'Assets','Fields of interest', 'Level of interest', 'NACE']].dropna().copy()

#Split catagorial data

levels = split_column(data, 'Level of interest', prefix = 'Level')
data.drop('Level of interest', axis = 1, inplace = True)
fields  = split_column(data , 'Fields of interest', prefix = 'Field')
data.drop('Fields of interest', axis = 1, inplace = True)

data = pd.concat( [data , levels, fields], axis = 1)


dependent = ['Degree', 'Strength', 'Betweenness', 'EV_linear', 'EV_log exp', 'EV_max', 'Hypercoreness']

lobbying = ['Members FTE']
financials = ['Assets']
#, 'Nb employees', 'Revenue']

country = ['TR Country']

sector = ['NACE']

level = list(levels.columns)
field = list(fields.columns)
# Group countries

country_groups = {
    'North America': ['CANADA', 'UNITED STATES'],
    'Central and South America': ['MEXICO', 'BRAZIL'],
    'North Europe': ['NORWAY', 'FINLAND', 'DENMARK', 'SWEDEN',  'LITHUANIA', 'LATVIA', 'ESTONIA', 'UNITED KINGDOM', 'IRELAND'],
    'South Europe': ['ITALY', 'SPAIN', 'PORTUGAL', 'GREECE', 'CROATIA', 'CYPRUS', 'SLOVENIA', 'GIBRALTAR', 'MALTA'],
    'East Europe': ['CZECH REPUBLIC', 'POLAND',   'ROMANIA', 'HUNGARY', 'BULGARIA', 'SLOVAKIA', 'KOSOVO', 'UKRAINE', 'RUSSIA, FEDERATION OF'],
    'West Europe': ['AUSTRIA', 'GERMANY', 'SWITZERLAND', 'BELGIUM', 'LUXEMBOURG', 'FRANCE', 'NETHERLANDS'],
    'East Asia': ['JAPAN', 'CHINA', 'HONG KONG', 'TAIWAN', 'KOREA, REPUBLIC OF'],
    'South Asia' : ['INDIA'],
    'Middle east' : ['UNITED ARAB EMIRATES', 'QATAR'],
    'Oeania' : [ 'SINGAPORE', 'AUSTRALIA','NEW ZEALAND'],
    'Other': ['SOUTH AFRICA', 'BERMUDA', 'GHANA', 'KAZAKHSTAN']
}
country_aff = {}
for region in country_groups.keys():
    for c in country_groups[region]:
        country_aff [c] = region

data['TR Country'] = data['TR Country'].apply(lambda x : country_aff[x])


binary = field + level
continuous = lobbying+ financials
nominal = country + sector

# Get columns with similar values
sim_val_cols = get_similar_value_cols(data, percent=80)
data = data.drop(columns = sim_val_cols)
field = list(set(data.columns) & set(field) )
level = list(set(data.columns) & set(level) )

# Remove outliers
threshold = 1 #percent
data = remove_outliers(data, 'TR Country', threshold)
data = remove_outliers(data, 'NACE', threshold)


# Correlation between cenrtalities
heatmap_corr (data,  dependent , annot = True)

# log tranform skewed continous variables
columns = list(get_skewed_columns(data[continuous +dependent]).index)
data = transform_skewed(data, columns,'log')


#X[level +country +financials + sector+ lobbying  ].to_csv(path + 'Programs/data_test_R.csv', index = False)

Y = data[dependent]


#heatmap_corr (data,  dependent , annot = True)
##



category_to_remove= {'Strength': {'Country' : 'East Europe', 'Sector' :'H - Transportation and storage'},
                'Closeness': {'Country' : 'East Europe', 'Sector' :'S - Other service activities'},
                'EV_max' : {'Country' :  'South Europe', 'Sector' : 'B - Mining and quarrying'},
                'Hypercoreness': {'Country' : 'North America' , 'Sector' :'H - Transportation and storage'}}


# Define variables for regression
targets = ['Strength', 'EV_linear', 'Hypercoreness']
variables = lobbying + financials + country + sector +level
basetable = data[variables + targets]

basetable, h_sector = one_hot_encod(basetable, sector, 'C - Manufacturing')
basetable, h_country = one_hot_encod(basetable, country, 'West Europe')
variables =  lobbying + financials + h_country + h_sector +level


#xy_heatmap_corr(basetable, field, h_sector)


# Normalize data
basetable = pd.DataFrame(preprocessing.StandardScaler().fit_transform(basetable) , columns = basetable.columns, index = basetable.index)



df_regressions = pd.DataFrame()
for target in  ['Strength', 'EV_linear', 'Hypercoreness']:
    target = 'Strength'
    y = basetable[target]
    X = basetable[variables]
    results = sm.OLS(y ,X.assign(const = 1)).fit()

    df = pd.DataFrame(results.params.round(3), columns = [target])
    df.loc[(0.01<results.pvalues) & (results.pvalues <0.05)] = df.astype(str) + "*"
    df.loc[(0.001<results.pvalues) & (results.pvalues <0.01)] = df.astype(str) + "**"
    df.loc[results.pvalues <0.001] = df.astype(str) + "***"
    df.loc[results.pvalues >0.05] = df.astype(str) +  ' '
    df_regressions = pd.concat([df_regressions, df], axis = 1)

#print(df_regressions.to_latex())
print(df_regressions)
print(results.summary())


## Analysis of variance

def pr_F(current_model , variables, tagret, basetable):
    X = basetable[variables]
    y = basetable[target]
    model = sm.OLS(y, X.assign(cont = 1)).fit()
    if model.df_model > current_model.df_model:
        anova_results = sm.stats.anova_lm(current_model, model )
    else:
        anova_results = sm.stats.anova_lm( model , current_model)
    return(anova_results['Pr(>F)'][1])


def next_best (current_variables, candidate_variables, taget, basetable):
    best_pr_F = 1
    best_variable = None

    X = basetable[current_variables]
    y = basetable[target]
    current_model = sm.OLS(y, X.assign(cont = 1)).fit()
    for v in candidate_variables:
        pr_F_v = pr_F( current_model , current_variables +[v], target, basetable)
        if pr_F_v < best_pr_F and pr_F_v < 0.05:
            best_pr_F = pr_F_v
            best_variable = v
    return best_variable

def next_worst (current_variables, target, basetable):
    worst_pr_F = 0
    worst_variable = None

    X = basetable[current_variables]
    y = basetable[target]
    current_model = sm.OLS(y, X.assign(cont = 1)).fit()
    for v in current_variables:

        variables = list( set(current_variables) - set([v]))
        pr_F_v = pr_F( current_model , variables, target, basetable)
        if pr_F_v > worst_pr_F and pr_F_v > 0.05 or np.isnan(pr_F_v) :
            worst_pr_F = pr_F_v
            worst_variable = v
    return worst_variable

def variables_to_add(current_variables, candidate_variables, target, basetable, threshold = 0.05):
    variables  = []

    X = basetable[current_variables]
    y = basetable[target]
    current_model = sm.OLS(y, X.assign(cont = 1)).fit()
    for v in candidate_variables:
        pr_F_v = pr_F( current_model , current_variables +[v], target, basetable)
        if pr_F_v < threshold:
            variables.append(v)
    return variables

def variables_to_remove(current_variables, target, basetable, threshold =0.05):

    variables =[]
    X = basetable[current_variables]
    y = basetable[target]
    current_model = sm.OLS(y, X.assign(cont = 1)).fit()
    for v in current_variables:
        red_variables = list( set(current_variables) - set([v]))
        pr_F_v = pr_F( current_model , red_variables, target, basetable)
        if pr_F_v > threshold or np.isnan(pr_F_v) :
            variables.append(v)
    return variables


def forward_selection(candidate_variables, target, basetable):
    current_variables = []
    nb_itt_max = len(candidate_variables)
    for i in range( nb_itt_max):
        next_var = next_best(current_variables, candidate_variables, target, basetable)
        if next_var is None:
            break
        current_variables = current_variables + [next_var]
        candidate_variables.remove(next_var)
    return(current_variables)

def backword_selection(candidate_variables, target, basetable):
    current_variables = lobbying + financials + h_country + h_sector
    nb_itt_max = len(current_variables)

    for i in range( nb_itt_max):
        next_var = next_worst(current_variables, target, basetable)
        if next_var is None:
            break
        current_variables.remove(next_var)
    return(current_variables)


    ##

# forward selection
target =  ['Strength']
basetable = data[lobbying + financials + country + sector + target+level]
basetable, h_sector = one_hot_encod(basetable, sector)
basetable, h_country = one_hot_encod(basetable, country)

candidate_variables = lobbying + financials + h_country + h_sector +level
current_variables = forward_selection(candidate_variables, target, basetable)
print(sm.OLS(basetable[target], basetable[current_variables].assign(const = 1)).fit().summary())

#backword selection

current_variables = backword_selection(candidate_variables, target, basetable)
print(sm.OLS(basetable[target] , basetable[current_variables].assign(const = 1)).fit().summary())

##

def stepwise_selection(candidate_variables, target, basetable):
    current_variables = []
    nb_itt_max = len(candidate_variables)
    while True:
        c = 0
        next_var = variables_to_add(current_variables, candidate_variables, target, basetable, threshold = teta_in)
        print(next_var)
        if next_var ==[]:
            c+=1
        else :
            current_variables = current_variables + next_var
            for v in next_var:
                candidate_variables.remove(v)
        next_var  = variables_to_remove(current_variables, target, basetable, threshold = teta_out)
        print(next_var)
        if next_var == []:
            c+=1
        else:
            for v in next_var:
                current_variables.remove(v)
                candidate_variables.append(v)

        if c == 2:
            break
    return(current_variables)



# Define variables for regression
targets = ['Strength', 'EV_linear', 'Hypercoreness']
variables = lobbying + financials + country + sector +level
basetable = data[variables + targets]

basetable, h_sector = one_hot_encod(basetable, sector)
basetable, h_country = one_hot_encod(basetable, country)
candidate_variables = lobbying + financials + h_country + h_sector + level


# Normalize data
basetable = pd.DataFrame(preprocessing.StandardScaler().fit_transform(basetable) , columns = basetable.columns, index = basetable.index)

#Stepwise selection
teta_in = 0.09
teta_out = 0.12
target =  ['EV_linear']
current_variables = stepwise_selection(candidate_variables, target, basetable)


print(sm.OLS(basetable[target] , basetable[current_variables].assign(const = 1)).fit().summary(slim = False ))

print(fr'$$\theta_i = {teta_in}, \theta_o ={teta_out}$$')
#print(sm.OLS(basetable[target] , basetable[current_variables].assign(const = 1)).fit().summary(slim = True ).as_latex())

##

# Define variables for regression
targets = ['Strength', 'EV_linear', 'Hypercoreness']
variables = lobbying + financials + country + sector +level
basetable = data[variables + targets]

basetable, h_sector = one_hot_encod(basetable, sector)
basetable, h_country = one_hot_encod(basetable, country)
variables = lobbying + financials + h_country + h_sector + level


# Normalize data
basetable = pd.DataFrame(preprocessing.StandardScaler().fit_transform(basetable) , columns = basetable.columns, index = basetable.index)


df_regressions = pd.DataFrame()
df_regressions.index = lobbying + financials + h_country + h_sector  +level


for target in  [['Strength'],['EV_linear'], ['Hypercoreness']]:
    candidate_variables = lobbying + financials + h_country + h_sector  +level
    y = basetable[target]
    variables = stepwise_selection(candidate_variables, target, basetable)

    X = basetable[variables]
    results = sm.OLS(y, X.assign(const = 1)).fit()
    print(results.rsquared)

    df = pd.DataFrame(results.params.round(3), columns = target)
    df.loc[(0.01<results.pvalues) & (results.pvalues <0.05)] = df.astype(str) + "*"
    df.loc[(0.001<results.pvalues) & (results.pvalues <0.01)] = df.astype(str) + "**"
    df.loc[results.pvalues <0.001] = df.astype(str) + "***"
    df.loc[results.pvalues >0.05] = df.astype(str) +  ' '
    df_regressions = pd.concat([df_regressions, df], axis = 1)

df_regressions.columns = df_regressions.columns.str.replace('_', ' ')
print(df_regressions)
print(df_regressions.to_latex(na_rep = ''))



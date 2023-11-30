import pandas as pd
import statsmodels.api as sm
path = '/home/azaiez/Documents/Cours/These/European Commission/'

Y = pd.read_csv(path + 'Programs/entities_centrality.csv', index_col = 'Name')
X = pd.read_csv(path+'Programs/MFA_coord.csv', index_col = 0)
X.index.names = ['Name']
#select the entities
Y = Y.loc[X.index]

y ='Degree'
#x = np.sort(list(levels))

Y = Y[y]


X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()


print(results.summary())
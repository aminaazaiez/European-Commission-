import hypernetx as hnx
import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import xgi

import numpy as np

from multiprocessing import Pool
import itertools


def XGI_2_nxBipartite (H : hnx.Hypergraph) :
    B = nx.Graph()

    B.add_nodes_from(H.nodes, bipartite='entity')
    B.add_nodes_from(H.edges , bipartite='meeting')
    for node in H.nodes:
        for edge in H.nodes.memberships(node):
            B.add_edge(node, edge)

    return B


## Strength
def strength (H : xgi.Hypergraph() ):
    cent = H.degree()
    return( pd.DataFrame({'Strength' : dict(cent).values()}, index = dict(cent).keys()) )

def degree(H : xgi.Hypergraph()):
    X =H.copy()
    X.merge_duplicate_edges()
    cent = X.degree()
    return( pd.DataFrame({'Degree' : dict(cent).values()}, index = dict(cent).keys()) )

def cardinality(H : xgi.Hypergraph()):
    cent = H.edges.size.asdict()
    return(pd.DataFrame({'Cardinality' : dict(cent).values()}, index = dict(cent).keys()) )



## Closeness
def closeness (H: xgi.Hypergraph):
    B = Hnx_2_nxBipartite(H)
    cent = bipartite.closeness_centrality(B ,bipartite.sets(B)[0])
    return( pd.DataFrame({'Closeness' : dict(cent).values()}, index = dict(cent).keys()) )

##Eigenvector centrality
def set_functions(mode):
    if mode == 'linear' :
        def f(x):
            return(x)
        def g(x):
            return (x)
        def psi(x):
            return(x)
        def phi(x):
            return(x)
        return(f,g,psi,phi)
    if mode == 'log exp':
        def f(x):
            return(x)
        def g(x):
            return (x**(1/2))
        def psi(x):
            return(np.exp(x))
        def phi(x):
            return(np.log(x))
        return(f,g,psi,phi)
    if mode == 'max':
        alpha = 10
        def f(x):
            return(x)
        def g(x):
            return (x)
        def psi(x):
            return(x**(1/alpha))
        def phi(x):
            return(x**alpha)
        return(f,g,psi,phi)


def eigenvector (H, mode = 'linear') :
    maxiter = 1000
    tol = 1e-6
    f,g,psi,phi = set_functions(mode)

    n = H.num_nodes
    m = H.num_edges
    x = np.ones(n)
    y = np.ones(m)

    I, node_dict, edge_dict = xgi.incidence_matrix(H, index=True)

    check = np.inf

    for iter in range(maxiter):

        u = np.sqrt(np.multiply(x, g(I @ f(y))))
        v = np.sqrt(np.multiply(y, psi(I.T @ np.nan_to_num(phi(x)))))

        new_x =  u / np.linalg.norm(u, 1)
        new_y = v / np.linalg.norm(v, 1)

        check = np.linalg.norm(new_x - x) + np.linalg.norm(new_y - y)
        if check < tol:
            break
        x = new_x.copy()
        y = new_y.copy()
    else:
        print("Iteration did not converge!")
    cent =  {node_dict[n]: new_x[n] for n in node_dict}
    cent.update({ edge_dict[e]: new_y[e] for e in edge_dict})
    return(pd.DataFrame({'EV_%s'%mode: cent.values() }, index = cent.keys()))
    # #
    #
    # B, idx , column  = xgi.incidence_matrix( H, index = True)
    # n,m = np.shape(B)
    #
    #
    # x0 = np.ones((n,1))
    # y0 = np.ones((m,1))
    # # x0 = np.random.rand(n,1)
    # # y0 = np.random.rand(m,1)
    #
    # for it in range(maxiter):
    #     if it%10 == 0:
    #         print(it)
    #
    #     u = np.sqrt(x0 * g(B @ f(y0)))
    #     v =np.sqrt( y0 * psi( np.transpose(B) @ np.nan_to_num(phi(x0))))
    #
    #     x = u / np.linalg.norm(u)
    #     y = v / np.linalg.norm(v)
    #
    #
    #     if np.linalg.norm(x - x0) + np.linalg.norm( y - y0) < tol :
    #         print('under tolerance value satisfied')
    #         x = np.reshape(x, n)
    #         y = np.reshape(y,m)
    #         return(pd.DataFrame({'EV_%s'%mode: x }, index = idx))
    #
    #     else :
    #         x0 = np.copy(x)
    #         y0 = np.copy(y)
    #
    # print('under tolerance value not satisfied')
    #
    # x = np.reshape(x, n)
    # y = np.reshape(y,m)
    # eigenvector_centrality = {idx[i] : x[i] for i in range(len(idx))}
    #
    # return(pd.DataFrame({'EV_%s'%mode: x }, index = idx))

## Betweenness Centrality

def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel_unipartite(B, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(B.nodes(), B.order() // node_divisor))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_subset,
        zip(
            [B] * num_chunks,
            node_chunks,
            [list(B)] * num_chunks,
            [True] * num_chunks,
            [None] * num_chunks,
        ),
    )

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c

def betweenness(H: hnx.Hypergraph):
    B = XGI_2_nxBipartite(H)
    nodes =  bipartite.sets(B)[1]
    top = set(nodes)
    bottom = set(B) - top
    n = len(top)
    m = len(bottom)
    s, t = divmod(n - 1, m)
    bet_max_top = (
        ((m**2) * ((s + 1) ** 2))
        + (m * (s + 1) * (2 * t - s - 1))
        - (t * ((2 * s) - t + 3))
    ) / 2.0
    p, r = divmod(m - 1, n)
    bet_max_bot = (
        ((n**2) * ((p + 1) ** 2))
        + (n * (p + 1) * (2 * r - p - 1))
        - (r * ((2 * p) - r + 3))
    ) / 2.0
    betweenness = betweenness_centrality_parallel_unipartite(B)
    for node in top:
        betweenness[node] /= bet_max_top
    for node in bottom:
        betweenness[node] /= bet_max_bot

    return pd.DataFrame({'Betweenness' : dict(betweenness).values()}, index = dict(betweenness).keys())

## Hypercoreness

def hypercoreness(core):
    c_m = { node : {} for node in core[2][1][0]}
    for m in core.keys():
        k_max = max(core[m].keys())
        for k in core[m].keys():
            if k != k_max :
                k_m_shell = core[m][k][0] - core[m][k+1][0]
            else :
                k_m_shell = core[m][k][0]

            for node in k_m_shell:
                c_m[node][m] = k/k_max


    R_i = {node : sum( [ c_m[node][m] for m in c_m[node].keys() ] ) for node in c_m.keys()}

    cent = pd.DataFrame({'Hypercoreness' : R_i.values()}, index = R_i.keys())
    return(cent)



## mettings' size attendence
def MSA (H : xgi.Hypergraph):
    edge_size = {}
    for node in H.nodes:
        E_i = H.nodes.memberships(node)
        edge_size[node] = sum ([ H.size (e) for e in E_i]) /len(E_i)

    return( pd.DataFrame({'MSA' : dict(edge_size).values()}, index = dict(edge_size).keys()))

## Diversity of meetings
from scipy.stats import entropy
def diversity(H : hnx.Hypergraph,  df ):
    feature  = 'Category of registration'
    df = EU.entities.dropna(subset = feature)
    membership = { orga : category for orga , category in zip ( df['Name'] , df[feature])}
    o= { agent : 0 for agent in list(df['Name'])}
    for e in H.edges():
        w_e = H.edges[e].weight
        d = H.size(e)
        orga_e = [node  for node in H.edges[e] if node in list(df['Name'])]
        c= Counter([ membership[agent] for agent in orga_e ])
        pk =[c[item]/d for item in c.keys()]
        h_e = entropy(pk)
        for agent in orga_e:
            o[agent] +=  w_e*  h_e
    return(o)

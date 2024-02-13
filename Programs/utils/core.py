import xgi
import hypernetx as hnx
import numpy as np
import itertools



def build_hypergraph(H: xgi.Hypergraph):
    X = H.copy()
    X.merge_duplicate_edges()

    return(X)

def restrict_m_hypergraph(X : xgi.Hypergraph, m: int , connected =False, intimate = False):
    ''' m : minimum size of hyperedge'''
    if intimate :
        e_remove = [edge for edge in X.edges if X.edges.size[edge] > m]
    else :
        e_remove = [edge for edge in X.edges if X.edges.size[edge] < m]

    X.remove_edges_from(e_remove)
    X.merge_duplicate_edges()
    still_2_remove = len(e_remove) > 0

    if connected :
        try:
            X.cleanup(isolates=False, singletons=False, multiedges=False, connected=True, relabel=False, in_place=True) # keep the larget connected component
        except:
            pass
    return(X, still_2_remove)

def restrict_k_hypergraph(X : xgi.Hypergraph, k: int, connected = False ):
    ''' k : minimum degree of nodes'''
    n_remove = [node for node in X.degree().keys() if X.degree()[node]< k]
    X.remove_nodes_from(n_remove)
    X.merge_duplicate_edges()
    still_2_remove = len(n_remove) > 0

    if connected :
        try:
            X.cleanup(isolates=False, singletons=False, multiedges=False, connected=True, relabel=False, in_place=True) # keep the larget connected component
        except:
            pass
    return(X , still_2_remove)



def core_decomposition(H : xgi.Hypergraph, connected = False, intimate = False) :
    X = build_hypergraph(H)
    if intimate :
        M = range( max(X.edges.size.asnumpy()) + 1 , 1 , -1) # m = edge size
    else:
        M = range(2 , max(X.edges.size.asnumpy()) + 1 ) # m = edge size


    K = range(max(X.degree().values()))
    core = { m : {} for m in M } # m-shell index

    for m in M: # loop for the m shells
        # For each m, start with the initial hypergraph restricted to the edges of size >= m

        k = 1
        X = build_hypergraph(H)
        while X.num_nodes > 0 : # loop for the k,m shell
            print(f'm ={m} , k = {k}')

            X , still_nodes_2_remove = restrict_m_hypergraph(X, m, connected, intimate)
            X , still_edges_2_remove = restrict_k_hypergraph(X, k, connected)
            # Store previous shell to compute the k,m shell at the end of the loop

            while  still_nodes_2_remove or still_edges_2_remove : # redo untill there are nither nodes nore edges that can be removed

                X , still_nodes_2_remove = restrict_m_hypergraph(X, m, connected, intimate)
                X , still_edges_2_remove = restrict_k_hypergraph(X, k, connected)

            if X.num_nodes > 0 :
                core[m][k] = [list(component) for component in xgi.connected_components(X)]

                k += 1

    return(core)

def k_m_core(H : hnx.Hypergraph, k, m):

    X = build_hypergraph(H)

    X , still_nodes_2_remove = restrict_m_hypergraph(X, m, )
    X , still_edges_2_remove = restrict_k_hypergraph(X, k, )

    while  still_nodes_2_remove or still_edges_2_remove : # redo untill there are nither nodes nore edges that can be removed

            X , still_nodes_2_remove = restrict_m_hypergraph(X, m, )
            X , still_edges_2_remove = restrict_k_hypergraph(X, k, )
    return(X)





def k_m_largest_cc_size(core,m , k):
    components = core[m][k]
    if len(components)>1:
        components.sort(key = len, reverse = True)
        return(len(components[0]) /( len(components[0])+ len(components[1]) ) )
    else :
        return(1)

def proportion_largest_cc_size(core):
    K = set(list (itertools.chain.from_iterable([ core[i].keys() for i in core.keys()])))
    M = core.keys()
    array = np.zeros( (len(M),len(K) ))
    for i,m in enumerate( M):
        for j,k in enumerate(K):
            try:
                array[i,j] = k_m_largest_cc_size(core, m,k)
            except :
                pass
    return(array)


def k_m_core_size(core):
    K = set(list (itertools.chain.from_iterable([ core[i].keys() for i in core.keys()])))
    M = core.keys()

    n_k_m =  np.zeros( (len(M),len(K) ))
    for i,m in enumerate( M):
        for j,k in enumerate(K):
            try:
                n_k_m[i,j] = sum([len(component) for component in core[m][k]])
            except :
                pass
    return(n_k_m)

def species_survival(core, m , k , node_set):
    k_m_core = core[m][k]
    nodes = set()
    for component in k_m_core:
        nodes = nodes.union(component)

    return( len( nodes & node_set ) )



def reverse_y_axis( A ):
    Y = np.shape(A)[0]
    X = np.shape(A)[1]

    array = np.zeros_like(A)
    for y in range(Y):
        for x in range(X):
            array [ Y - y-1,x]=  A[ y, x]
    return(array)


def exiting_shell(core) :
    x_min = min(core.keys())
    y_min = min(core[x_min].keys())
    c_m = { node : {} for node in core[x_min][y_min][0]}
    for x in core.keys():
        y_max = max(core[x].keys())
        for y in core[x].keys():
            if y != y_max :
                x_y_shell = core[x][y][0] - core[x][y+1][0]
            else :
                x_y_shell = core[x][y][0]

            for node in x_y_shell:
                c_m[node][x] = y
    return(c_m)



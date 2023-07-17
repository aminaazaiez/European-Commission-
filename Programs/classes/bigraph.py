from networkx.algorithms import bipartite

from multiprocessing import Pool
import itertools

import networkx as nx


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel_unipartite(self, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(self.nodes(), self.order() // node_divisor))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_subset,
        zip(
            [self] * num_chunks,
            node_chunks,
            [list(self)] * num_chunks,
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

def betweenness_centrality_parallel(self, nodes):
    
    top = set(nodes)
    bottom = set(self) - top
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
    betweenness = betweenness_centrality_parallel_unipartite(self)
    for node in top:
        betweenness[node] /= bet_max_top
    for node in bottom:
        betweenness[node] /= bet_max_bot
    return betweenness

bipartite.betweenness_centrality_parallel = betweenness_centrality_parallel


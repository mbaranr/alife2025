import networkx as nx
import numpy as np

from grow.reservoir import Reservoir


def _to_undirected_nx(res: Reservoir) -> nx.Graph:
    A = np.asarray(res.A)
    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    rows, cols = np.nonzero(A)
    for u, v in zip(rows.tolist(), cols.tolist()):
        if u == v:
            continue
        G.add_edge(u, v)

    return G


def _gini(xs) -> float:
    x = np.array(xs, dtype=float)
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def is_lin(res: Reservoir) -> bool:
    """
    Treat graph as undirected.
    Split into connected components.
    A component is 'linear' if diameter == n_c - 1.
    Graph is 'linear' if all components are linear.
    """
    G = _to_undirected_nx(res)
    for nodes in nx.connected_components(G):
        H = G.subgraph(nodes)
        n_c = H.number_of_nodes()
        # diameter for a 0/1-node component is defined as 0 in nx
        diam = nx.diameter(H) if n_c > 0 else 0
        if diam != n_c - 1:
            return False
    return True


def is_loose(res: Reservoir) -> bool:
    """
    Not linear.
    Mean normalized betweenness:  g(v) / ((n_c-1)(n_c-2)) > 0.01
    Mean reciprocal closeness:    1 / C(v) > 0.07 * n_c
    Degree Gini coefficient < 0.1 (no hubs)
    Graph is 'loosely stranded' if all components satisfy all three.
    """
    if is_lin(res):
        return False

    G = _to_undirected_nx(res)

    for nodes in nx.connected_components(G):
        H = G.subgraph(nodes)
        n_c = H.number_of_nodes()

        # betweenness condition
        if n_c >= 3:
            bc = nx.betweenness_centrality(H, normalized=False)
            denom = (n_c - 1) * (n_c - 2)
            mean_norm_bc = float(np.mean([v / denom for v in bc.values()]))
            if not (mean_norm_bc > 0.01):
                return False

        # reciprocal closeness condition (avg distance)
        # closeness_centrality returns C(v) = (n_c-1) / sum(dist(v,*))
        # so 1/C(v) is proportional to average distance.
        if n_c >= 2:
            cc = nx.closeness_centrality(H)
            inv_cc = []
            for v, c in cc.items():
                if c <= 0:
                    inv_cc.append(float("inf"))
                else:
                    inv_cc.append(1.0 / c)
            mean_inv_cc = float(np.mean(inv_cc))
            if not (mean_inv_cc > 0.07 * n_c):
                return False

        # degree Gini (no hubs)
        degs = [d for _, d in H.degree()]
        if _gini(degs) >= 0.1:
            return False

    return True


def res_type(res: Reservoir) -> str:
    if is_lin(res):
        return "lin"
    if is_loose(res):
        return "loose"
    return "misc"
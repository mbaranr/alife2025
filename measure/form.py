import networkx as nx
import numpy as np

from grow.reservoir import Reservoir


def _to_undirected_nx(res: Reservoir) -> nx.Graph:
    edges = res.to_edgelist()

    G = nx.Graph()
    G.add_nodes_from(range(res.size()))
    G.add_edges_from((int(u), int(v)) for u, v, _ in edges if u != v)

    return G


def is_lin(res: Reservoir, thr: int=3) -> bool:
    """
    Undirected. Each connected component is 'linear' if diameter == n_c - 1.
    """
    G = _to_undirected_nx(res)
    for nodes in nx.connected_components(G):
        H = G.subgraph(nodes)
        n_c = H.number_of_nodes()

        if n_c == 1:
            continue

        diam = nx.diameter(H) if n_c > 0 else 0
        
        degs = np.array([d for _, d in H.degree()])
        num_deg = np.sum(degs > 2)

        if diam != n_c - 1 and num_deg > thr:
            return False
        
    return True


def is_loose(res: Reservoir, bct: float=0.01, cct: float=0.07) -> bool:
    if is_lin(res):
        return False

    G = _to_undirected_nx(res)

    for nodes in nx.connected_components(G):
        H = G.subgraph(nodes)
        n_c = H.number_of_nodes()

        if n_c == 1:
            continue
        
        if n_c >= 2:
            cc = nx.closeness_centrality(H)
            if np.mean([1 / c for c in cc.values()]) <= cct * n_c:
                return False
            
        if n_c >= 3:
            bc = nx.betweenness_centrality(H, normalized=True)
            mean_bc = np.mean(list(bc.values()))
            if mean_bc <= bct:
                return False

        degs = np.array([d for _, d in H.degree()])
        if degs.max() >= 3 * degs.mean():
            return False
        
    return True


def res_type(res: Reservoir) -> str:
    if is_lin(res):
        return "Linear"
    if is_loose(res):
        return "Loosely"
    return "Other"
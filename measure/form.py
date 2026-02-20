import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from grow.reservoir import Reservoir


def mean_closeness(H):
    cc = nx.closeness_centrality(H)
    return float(np.mean(list(cc.values())))

def mean_betweenness(H):
    bc = nx.betweenness_centrality(H)
    return float(np.mean(list(bc.values())))

def diameter(H):
    return float(nx.diameter(H))


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
    G = _to_undirected_nx(res.no_selfloops())
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


def res_kmeans(
    k,
    reservoirs,
    component_fns,
    comp_thresh=0.2,
    scaler_cls=StandardScaler,
    random_state=0,
    n_init=20,
):
    if not (0 <= comp_thresh <= 1):
        raise ValueError("comp_thresh must be in [0, 1].")

    kept_indices = []
    feats_out = []

    for idx, res in enumerate(reservoirs):
        G = _to_undirected_nx(res)
        min_nodes = comp_thresh * res.size()

        comp_feats = []
        for nodes in nx.connected_components(G):
            H = G.subgraph(nodes).copy()
            if H.number_of_nodes() >= min_nodes:
                vals = []
                for fn in component_fns:
                    try:
                        v = fn(H)
                    except Exception:
                        v = np.nan
                    vals.append(v)
                comp_feats.append(vals)

        if not comp_feats:
            continue

        comp_feats = np.asarray(comp_feats, dtype=float)
        feat_mean = np.nanmean(comp_feats, axis=0)

        if np.isnan(feat_mean).any():
            continue

        kept_indices.append(idx)
        feats_out.append(feat_mean)

    X = np.asarray(feats_out, dtype=float)

    scaler = scaler_cls()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X_scaled)

    clusters = [[] for _ in range(k)]
    for local_i, lab in enumerate(labels):
        orig_idx = kept_indices[local_i]
        clusters[int(lab)].append(orig_idx)

    return clusters, labels, kept_indices, X, km, scaler
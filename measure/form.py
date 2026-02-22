import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from grow.reservoir import Reservoir


def norm_diameter(H):
    return float(nx.diameter(H)/len(H.nodes))

def p90_betweenness(H):
    bc = nx.betweenness_centrality(H)
    vals = np.array(list(bc.values()), dtype=float)
    if vals.size == 0:
        return np.nan
    return float(np.percentile(vals, 90))

def median_closeness(H):
    cc = nx.closeness_centrality(H)
    vals = np.array(list(cc.values()), dtype=float)
    if vals.size == 0:
        return np.nan
    return float(np.median(vals))


def _to_undirected_nx(res: Reservoir) -> nx.Graph:
    edges = res.to_edgelist()

    G = nx.Graph()
    G.add_nodes_from(range(res.size()))
    G.add_edges_from((int(u), int(v)) for u, v, _ in edges if u != v)

    return G


def res_kmeans(
    k,
    reservoirs,
    component_fns,
    scaler_cls=StandardScaler,
    random_state=0,
    n_init=20,
):
    kept_indices = []
    feats_out = []

    for idx, res in enumerate(reservoirs):
        G = _to_undirected_nx(res)
        n_res = res.size()
        if n_res <= 0:
            continue

        comp_feats = []
        comp_w = []

        for nodes in nx.connected_components(G):
            H = G.subgraph(nodes).copy()
            w = H.number_of_nodes() / n_res  # normalized component size weight

            vals = []
            for fn in component_fns:
                try:
                    v = fn(H)
                except Exception:
                    print(f"Error computing component feature, setting to NaN: {fn.__name__} on component with nodes {nodes}")
                    v = np.nan
                vals.append(v)

            comp_feats.append(vals)
            comp_w.append(w)

        if not comp_feats:
            continue

        comp_feats = np.asarray(comp_feats, dtype=float)      # (n_comp, n_feat)
        comp_w = np.asarray(comp_w, dtype=float)              # (n_comp,)

        # weighted average per feature, renormalizing weights
        feat_mean = np.empty(comp_feats.shape[1], dtype=float)
        for j in range(comp_feats.shape[1]):
            col = comp_feats[:, j]
            mask = ~np.isnan(col)
            if not np.any(mask):
                feat_mean[j] = np.nan
                continue
            wj = comp_w[mask]
            s = wj.sum()
            if s == 0:
                feat_mean[j] = np.nan
                continue
            feat_mean[j] = np.dot(col[mask], wj) / s

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
# Adapted from Waldegrave et al. (2024): https://doi.org/10.1162/isal_a_00734
# Copyright (c) 2024, Riversdale Waldegrave


import numpy as np
import matplotlib.pyplot as plt

import graph_tool.all as gt
from scipy.sparse.csgraph import connected_components

from multiset import FrozenMultiset
from wrapt_timeout_decorator.wrapt_timeout_decorator import timeout


class GraphDef(object):

    def __init__(self, A: np.ndarray, S: np.ndarray):
        self.A = A      # N x N
        self.S = S      # N x S
        self.n_states = S.shape[1]

    def __str__(self) -> str:
        return f"Graph with {self.size()} nodes and {self.num_edges()} edges"

    def size(self) -> int:
        return self.A.shape[0]
    
    def num_edges(self) -> int:
        return np.sum(self.A)

    def connectivity(self) -> float:
        if self.size()>0:
            return self.num_edges() / self.size()**2
        else:
            return 0 # or could do np.nan?
    
    def get_neighbourhood(self, noise: float = 0.0) -> np.ndarray:
        """
        noise interpreted as exploration rate in [0,1]:
        - with probability `noise`, each existing edge is dropped
        - with probability `noise * density`, each non-edge is added (roughly keeps scale reasonable)
        """
        A = self.A

        if noise > 0:
            # work on a perturbed adjacency copy
            A = A.copy().astype(np.int8)
            n = A.shape[0]

            # drop existing edges
            drop_mask = (A == 1) & (np.random.random((n, n)) < noise)
            A[drop_mask] = 0

            # add some non-edges (scaled so it doesn't explode in dense graphs)
            density = A.mean() if n > 0 else 0.0
            add_p = noise * density
            add_mask = (A == 0) & (np.random.random((n, n)) < add_p)
            A[add_mask] = 1

        c_in = A.T @ self.S
        c_out = A @ self.S
        G = np.hstack([self.S, c_in, c_out])

        row_max = np.max(np.abs(G), axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        return G / row_max
    
    def to_edgelist(self) -> np.ndarray:
        """
        Returns an E*3 numpy array in the format
        source, target, weight (1s in this case)
        """
        es = np.nonzero(self.A)
        edge_list = np.array([es[0], es[1], self.A[es]]).T
        return edge_list
    
    def states_1d(self) -> list[int]:
        """
        Converts states out of one-hot encoding into a list of ints.
        eg. [[1,0,0,0],[0,0,1,0],[0,0,0,1]] -> [0,2,3]
        """
        return np.argmax(self.S, axis=1).tolist()
    
    def state_hash(self) -> int:
        """
        Returns a hash of all the nieghbourhood state info.
        This is good as a preliminary isomorphism check (if
        two hashes are not the same then the graphs are definitely
        different).
        """
        return hash(FrozenMultiset(map(tuple, self.get_neighbourhood().tolist())))

    @timeout(5, use_signals=False)
    def is_isomorphic(self, other: "GraphDef") -> bool:
        """
        Checks if this graph is isomorhpic with another, conditional on node states.
        The decorator makes this function raise a timeout error if it takes longer than 5 seconds.
        """
        ne1 = self.num_edges()
        ne2 = other.num_edges()
        if ne1!=ne2:
            return False
        if ne1==0:
            # neither have any edges: structure doesn't matter so just check states match
            s1 = self.states_1d()
            s2 = other.states_1d()
            s1.sort()
            s2.sort()
            return s1==s2 
        gt1, gt2 = self.to_gt(), other.to_gt()
        # despite function name, subgraph=False does whole graph isomorphism
        # vertex_label param is used to condition the isomorphism on node state.
        mapping = gt.subgraph_isomorphism(gt1, gt2, 
                                          vertex_label=[gt1.vp.state, gt2.vp.state], 
                                          subgraph=False)
        # is isomorphic if at least one mapping was found
        return False if len(mapping)==0 else True
    
    def draw_gt(self, draw_edge_wgt: bool=False,
                pos: gt.VertexPropertyMap=None,
                interactive: bool=False,
                **kwargs) -> gt.VertexPropertyMap:
        """
        Draws a the graph using the graph-tool library 
        Relies on node and edge measure set by to_gt()
        Returns the node positions, which can then be passed in at the next
        call so that original nodes don't move to much if you are adding more etc.
        NB use output=filename to write to a file.
        """
        if self.size() == 0:
            print("Empty graph - can't draw")
            return None
        
        g = self.to_gt(pos=pos, pp=True)

        # edge weights if enabled
        edge_pen_width = gt.prop_to_size(g.ep.wgt, mi=1, ma=7) if draw_edge_wgt else None

        # draw the graph
        if interactive:
            gt.interactive_window(
                g, pos=g.vp['pos'], vertex_fill_color=g.vp['plot_color'],
                vertex_color=g.vp['outline_color'],
                edge_pen_width=edge_pen_width,
                edge_color=g.ep['edge_color'],
                **kwargs
            )
        else:
            gt.graph_draw(
                g, pos=g.vp['pos'], vertex_fill_color=g.vp['plot_color'],
                vertex_color=g.vp['outline_color'],
                edge_pen_width=edge_pen_width,
                edge_color=g.ep['edge_color'],
                **kwargs
            )

    def to_gt(self, basic: bool=False, 
              pos: gt.VertexPropertyMap=None,
              pp: bool=False) -> gt.Graph:
        """
        Converts it to a graph-tool graph.
        Good for visualisation and isomorphism checks.
        Nodes are coloured by state.
        Use basic=True if you just want the graph structure
        """
        n_nodes = self.size()
        edge_list = self.to_edgelist()
        g = gt.Graph(n_nodes)
        g.add_edge_list(edge_list, eprops=[("wgt", "double")])

        if not basic:
            # assign node states as an internal property
            states = g.new_vertex_property('int', self.states_1d())
            g.vp['state'] = states

            # helper to set IO positions and colors
            if pp:
                self._pp(g, pos=pos)            
        return g
    
    def _pp(self, g: gt.Graph, 
           pos: gt.VertexPropertyMap = None) -> gt.VertexPropertyMap:
        """
        Pretty prints the graph.
        """
        # assign colors based on states
        states_1d = self.states_1d()
        cmap = plt.get_cmap('viridis', self.n_states + 1)
        state_colors = cmap(states_1d)
        g.vp['plot_color'] = g.new_vertex_property('vector<double>', state_colors)
        pos = g.new_vertex_property("vector<double>")
        g.vp['pos'] = pos

    def no_selfloops(self) -> "GraphDef":
        """
        Returns a copy of the graph in which all self-loops have been removed
        """
        out_A = self.A.copy()
        # set values on the diagonal to zero
        out_A[np.eye(out_A.shape[0], dtype=np.bool_)] = 0 
        return GraphDef(out_A, self.S.copy())
    
    def get_components(self) -> tuple[np.ndarray]:
        """
        Returns a number for each node indicating which component
        it is part of.
        eg. [1,1,2,2,1] means nodes 0,1,4 form one connected component
        and nodes 2&3 form another.
        """
        # undirected for this purpose.
        _, cc = connected_components(self.A, directed=False)
        # count nodes in each component (will be sorted by component label 0->n_components)
        _, counts = np.unique(cc, return_counts=True)
        return cc, counts
    
    def get_largest_component_frac(self) -> float:
        """
        Returns the size of the largest component as a fraction of 
        the total number of nodes in the graph.
        """
        if self.size()==0:
            return 0
        else:
            _, component_sizes = self.get_components()
            return np.max(component_sizes) / self.size()

    def copy(self):
        return GraphDef(np.copy(self.A), np.copy(self.S))

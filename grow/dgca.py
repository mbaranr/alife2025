# Adapted from Waldegrave et al. (2024): https://doi.org/10.1162/isal_a_00734
# Copyright (c) 2024, Riversdale Waldegrave


import numpy as np
from grow.reservoir import Reservoir


def stable_sigmoid(x):
    """
    Numerically stable sigmoid function.
    """
    x = np.clip(x, -500, 500)  # prevent very large positive/negative values
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def onehot(x: np.ndarray):
    """
    Helper function to on hot encode an array x.
    """
    tf = x == np.max(x, axis=1, keepdims=True)
    return tf.astype(int)


class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.random.uniform(-1, 1, (layer_sizes[i + 1],)))

    def forward(self, x):
        """
        Perform a forward pass through the MLP with stable sigmoid activation.
        """
        for i in range(len(self.weights) - 1):
            x = stable_sigmoid(np.dot(x, self.weights[i]) + self.biases[i])
        output = np.dot(x, self.weights[-1]) + self.biases[-1]
        return output

    def get_parameters(self):
        """
        Get the parameters of the MLP.
        """
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        """
        Set the parameters of the MLP.
        """
        assert len(weights) == len(self.weights), "Mismatch in number of weight matrices."
        assert len(biases) == len(self.biases), "Mismatch in number of bias vectors."
        self.weights = weights
        self.biases = biases


class DGCA(object):

    # kronocker product matrices
    Q_M = [[1,0], 
        [0,0]]

    Q_F = [[0,1], 
        [0,0]]

    Q_B = [[0,0], 
        [1,0]]

    Q_N = [[0,0], 
        [0,1]]

    def __init__(self, hidden_size=None, n_states: int=None, noise: float=0.0):
        if not n_states:
            return
        self.action_mlp = MLP(layer_sizes=[3 * n_states, hidden_size, 15] if hidden_size else [3 * n_states, 15])   # action MLP
        self.state_mlp = MLP(layer_sizes=[3 * n_states, n_states] if hidden_size else [3 * n_states, n_states])     # state SLP
        
        self.noise = noise

    def update_action(self, res: Reservoir):
        """
        First MLP.
        """
        G = res.get_neighbourhood(noise=self.noise)
        D = self.action_mlp.forward(G)   # N x 15

        # one hot in sections
        K = np.hstack((onehot(D[:,0:3]), onehot(D[:,3:7]), onehot(D[:,7:11]), onehot(D[:,11:15])))
        K = K.T 

        # action choices
        remove = K[0,:]
        noaction = K[1,:]
        divide = K[2,:]
        
        remove[:res.input_nodes+res.output_nodes] = 0    # I/O nodes

        keep = np.hstack((np.logical_not(remove), divide)).astype(bool)
        
        # new node wiring
        none_f, k_fi, k_fa, k_ft = K[3, :], K[4, :], K[5, :], K[6, :]
        none_b, k_bi, k_ba, k_bt = K[7, :], K[8, :], K[9, :], K[10,:]
        none_n, k_ni, k_na, k_nt = K[11,:], K[12,:], K[13,:], K[14,:]

        none_all =  np.hstack((np.zeros((res.size())), np.logical_and(none_f, none_b, none_n))).astype(bool)
        keep = np.logical_and(keep, np.logical_not(none_all))

        I = np.eye(res.size())

        A, S = res.A, res.S
        A_new = np.kron(self.Q_M, A) \
            + np.kron(self.Q_F, (I @ np.diag(k_fi) + A @ np.diag(k_fa) + A.T @ np.diag(k_ft))) \
            + np.kron(self.Q_B, (np.diag(k_bi) @ I + np.diag(k_ba) @ A + np.diag(k_bt) @ A.T)) \
            + np.kron(self.Q_N, (np.diag(k_ni) @ I + np.diag(k_na) @ A + np.diag(k_nt) @ A.T))
        
        # keep only the nodes we need
        A_new = A_new[keep,:][:,keep]

        # duplicate relevant cols of state matrix
        S_new = np.vstack((S, S))
        S_new = S_new[keep,:]

        return Reservoir(A_new, S_new, res.input_nodes, res.output_nodes).no_islands()
       
    def update_state(self, res: Reservoir):
        """
        Second SLP.
        """
        G = res.get_neighbourhood(noise=self.noise)
        D = self.state_mlp.forward(G)  # N x S
        return Reservoir(res.A, onehot(D), res.input_nodes, res.output_nodes)

    def step(self, res: Reservoir):
        """
        Pass through both MLPs.
        """
        pre = self.update_action(res)
        post = self.update_state(pre)
        return post
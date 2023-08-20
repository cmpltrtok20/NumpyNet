import numpy as np
from NumpyNet.activations import acts, eps


class ConnLayer(object):

    def __init__(self, netwrok, n_neurons, n_inputs, act):
        self.network = netwrok
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.coef = np.random.randn(n_inputs, n_neurons).astype(np.float32)
        self.bias = np.zeros([1, n_neurons], dtype=np.float32)
        self.act = acts[act]

    def trainable_vars(self):
        return self.coef.size + self.bias.size

    def __call__(self, inputs, *args, **kwargs):
        self.inputs = inputs
        m, n = inputs.shape
        self.X = inputs
        self.W = self.coef
        Z = np.dot(self.X, self.W) + self.bias
        A = self.act[0](Z)
        self.outputs = A
        return A

    def backward(self, delta):
        A = self.outputs
        m = len(self.X)
        dz = delta * self.act[1](A)  # chained diff
        self.delta_bias = dz.sum(axis=0) / m
        self.delta_coef = np.dot(self.X.T, dz) / m
        self.delta = np.dot(dz, self.W.T)  # chained diff
        return self.delta

    def step(self):
        self.bias -= self.network.lr * self.delta_bias
        self.coef -= self.network.lr * self.delta_coef

    def save_weights(self, vars):
        vars = np.append(vars, self.bias)
        vars = np.append(vars, self.coef)
        return vars

    def load_weights(self, vars, pos):
        # for bias vector
        pos_start = pos
        pos += self.n_neurons
        self.bias = vars[pos_start:pos].reshape(1, self.n_neurons)
        # for coef matrix
        pos_start = pos
        pos += self.n_neurons * self.n_inputs
        self.coef = vars[pos_start:pos].reshape(self.n_inputs, self.n_neurons)
        return pos


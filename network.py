import numpy as np
from NumpyNet.activations import acts, eps
from NumpyNet.layers.conn import ConnLayer

eps = 1e-20


def softmax_foward(inputs):
    exp = np.exp(inputs)
    exp_sum = exp.sum(axis=-1).reshape(*inputs.shape[:-1], 1)
    outputs = exp / (exp_sum + eps)
    return outputs


class NumpyNet(object):

    def __init__(self, conf):
        networkConf = conf['network']
        self.lr = networkConf['lr']
        n_inputs = networkConf['n_inputs']
        layersConf = conf['layers']
        self.n_layers = len(layersConf)

        self.layers = []
        n_all_vars = 0
        self.bin = False
        for i, layer in enumerate(layersConf):
            n_neurons = layer['n_outputs']
            act = layer['act']
            if i == self.n_layers - 1:
                self.n_cls = n_neurons
                if 1 == n_neurons:
                    self.bin = True
                if not self.bin:
                    if act != 'softmax':
                        raise Exception('Currently only softmax is supported as the last activation in multiple classifiction')
                else:
                    if act != 'sigmoid':
                        raise Exception('Currently only sigmoid is supported as the last activation in binary classifiction')
            layer = ConnLayer(self, n_neurons, n_inputs, act)
            n_vars = layer.trainable_vars()
            n_all_vars += n_vars
            print(f'{n_inputs} X {n_neurons} {n_vars}')
            self.layers.append(layer)
            n_inputs = n_neurons
        print(f'All vars: {n_all_vars}')

    def save_weights(self, path):
        print(f'Saving to {path}')
        vars = np.array([], dtype=np.float32)
        for layer in self.layers:
            vars = layer.save_weights(vars)
        np.save(path, vars, allow_pickle=False)
        print(f'Saved {len(vars)}')

    def load_weights(self, path):
        print('Loading ...')
        vars = np.load(path)
        print('vars', vars.shape)
        pos = 0
        for layer in self.layers:
            print('pos', pos)
            pos = layer.load_weights(vars, pos)
        print('pos', pos)
        print('Loaded.')

    def __call__(self, inputs, *args, **kwargs):
        for layer in self.layers:
            layer(inputs)
            inputs = layer.outputs
        outputs = layer.outputs
        if not self.bin:
            outputs = softmax_foward(outputs)
        self.outputs = outputs
        return self.outputs

    def cost(self, A, Y):
        m = len(Y)
        if self.bin:
            j = np.dot(Y.T, np.log(A + eps)) + np.dot((1.0 - Y.T), np.log(1.0 - A + eps))
            j = j[0][0] / -m
        else:
            j = (Y * np.log(A)).sum(axis=-1).sum()
            j = j / -m
        return j

    def metrics(self, Y, A, thresh=0.5):
        m = len(Y)
        if self.bin:
            Y = Y.astype(np.float32)
            ones = np.ones_like(Y, dtype=np.int64)
            TP = ones[(A > thresh) & (Y > thresh)].sum()
            TN = ones[(A < thresh) & (Y < thresh)].sum()
            FP = ones[(A > thresh) & (Y < thresh)].sum()
            FN = ones[(A < thresh) & (Y > thresh)].sum()
            acc = (TP + TN) / m
            recall = TP / (TP + FN + 1e-10)
            precision = TP / (TP + FP + 1e-10)
            f1 = (2 * recall * precision) / (recall + precision + 1e-10)
            return dict(
                acc=acc,
                recall=recall,
                precision=precision,
                f1=f1
            )
        else:
            acc = (Y.argmax(axis=-1) == A.argmax(axis=-1)).astype(np.int64).sum() / m
            return dict(
                acc=acc
            )

    def backward(self, Y):
        A = self.outputs
        if self.bin:
            delta = (A - Y) / (A * (1.0 - A) + eps)
        else:
            delta = A - Y
        for i, layer in enumerate(self.layers[::-1]):
            delta = layer.backward(delta)

    def step(self):
        for i, layer in enumerate(self.layers):
            layer.step()

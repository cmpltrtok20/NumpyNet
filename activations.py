import numpy as np

eps = 1e-20


def linear_forward(inputs):
    return inputs.copy()


def linear_backward(output):
    return np.ones_like(output, dtype=np.float32)


def sigmoid_forward(inputs):
    return 1. / (1. + np.exp(-inputs + eps))


def sigmoid_backward(output):
    return output * (1. - output)


def relu_forward(inputs):
    outputs = inputs.copy()
    outputs[inputs <= 0.] = 0.
    return outputs


def relu_backward(output):
    delta = np.zeros_like(output, dtype=np.float32)
    delta[output > 0.] = 1.0
    return delta


def leaky_forward(inputs):
    outputs = inputs.copy()
    outputs[inputs <= 0.] = 0.1 * inputs
    return outputs


def leaky_backward(output):
    delta = np.full(0.1, output.shape, dtype=np.float32)
    delta[output > 0.] = 1.0
    return delta


acts = dict(
    linear=(linear_forward, linear_backward),
    softmax=(linear_forward, linear_backward),  # softmax is handled by the network, not the activation
    sigmoid=(sigmoid_forward, sigmoid_backward),
    relu=(relu_forward, relu_backward),
    leaky=(leaky_forward, leaky_backward),
)
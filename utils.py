import numpy as np


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) - np.exp(-z))


def dtanh(z):
    return 1 - tanh(z) ** 2


def softmax(x, derivative=False):
    x_safe = x + 1e-12
    f = np.exp(x_safe) / np.sum(np.exp(x_safe))

    if derivative:  # Return the derivative of the function evaluated at x
        return 1  # We will not need this one
    else:  # Return the forward pass of the function at x
        return f

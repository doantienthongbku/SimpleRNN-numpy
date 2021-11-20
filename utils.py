import numpy as np


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) - np.exp(-z))


def dtanh(z):
    return 1 - tanh(z) ** 2


def softmax(z):
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    A = e_z / e_z.sum(axis=0)
    return A


def dsoftmax(z):
    # return (softmax(z + eps) - softmax(z - eps)) / (2 * eps)
    Sz = softmax(z)
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D

import numpy as np
from scipy.special import expit


def L2_reg(lambda_, w1, w2):
    """Compute L2-regularization cost"""
    return (lambda_ / 2.0) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))


def L1_reg(lambda_, w1, w2):
    """Compute L1-regularization cost"""
    return (lambda_ / 2.0) * (np.abs(w1).sum() + np.abs(w2).sum())


def cross_entropy(output, y_target):
    return - np.sum(np.log(output) * (y_target), axis=1)


def sigmoid(z):
    # return 1.0 / (1.0 + np.exp(-z))
    return expit(z)


def sigmoid_gradient(z):
    """Compute gradient of the logistic function"""
    sg = sigmoid(z)
    return sg * (1 - sg)

def softmax(z):
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

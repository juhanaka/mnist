from __future__ import division
import numpy as np


def sigmoid(scalar_or_array):
    return 1 / (1 + np.exp(-scalar_or_array))


def hypothesis(theta, X):
    return sigmoid(np.dot(X, theta))


def cost(theta, y, X):
    m = y.size
    left_side = -1 * y * np.log(hypothesis(theta, X))
    right_side = (1 - y) * np.log(1 - hypothesis(theta, X))
    ans_vector = left_side - right_side
    return (1 / m) * np.sum(ans_vector)


def gradient(theta, y, X):
    m = y.size
    ans = (1 / m) * np.dot((hypothesis(theta, X) - y).T, X)
    return ans.T


def gradient_descent(theta, y, X, alpha, n):
    theta_ = theta.copy()
    costs = np.zeros([n, 1])
    for i in range(n):
        theta_ -= alpha * gradient(theta_, y, X)
        costs[i] = cost(theta_, y, X)
    return theta_, costs

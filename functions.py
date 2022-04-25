import numpy as np


def gaussian(x, t):
    return np.exp(-np.square(x * t))


def euclidean_distance(x, y):
    sum = np.sum(np.square(x - y))
    return np.sqrt(sum)


import numpy as np
import sys

from functions import gaussian, euclidean_distance


class Network:
    def __init__(self, input_shape, shape, metric=euclidean_distance,
                 neighbourhood_func=gaussian):
        self.learning_rate = None
        self.n_epochs = None
        self.metric = metric
        self.neighbourhood_func = neighbourhood_func
        self.input_shape = input_shape
        self.shape = shape
        self.weights = None
        self.neurons = None
        self._initialize_neurons()
        self._initialize_weights()

    def _initialize_neurons(self):
        self.neurons = np.random.uniform(-1, 1, size=self.shape)

    def _initialize_weights(self):
        self.weights = np.random.uniform(-1, 1, size=[self.input_shape,
                                                      self.shape[0],
                                                      self.shape[1]])

    def decay_function(self, t):
        return np.exp(-t / self.n_epochs)

    def neighbourhood_weights(self, n1, n2, t):
        distance = self.metric(n1, n2)
        return self.neighbourhood_func(distance, t)

    def find_closest_neighbour(self, x):
        min_distance = sys.maxsize
        i_min = 0
        j_min = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                distance = self.metric(self.weights[:, i, j], x)
                if distance < min_distance:
                    min_distance = distance
                    i_min = i
                    j_min = j

        return i_min, j_min

    def fit(self, data, n_epochs):
        self.n_epochs = n_epochs

        for t in range(n_epochs):
            np.random.shuffle(data)
            for x in data:
                # TODO
                i_min, j_min = self.find_closest_neighbour(x)
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        n1 = np.array([i_min, j_min])
                        n2 = np.array([i, j])
                        delta_weights = self.neighbourhood_weights(n1, n2, t) \
                                        * self.decay_function(t) * (
                                                x - self.weights[:, i, j])
                        self.weights[:, i, j] += delta_weights

    def visualise(self):
import numpy as np


class ReLULayer(object):
    def __init__(self, name):
        super(ReLULayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        return np.maximum(0, X)

    def delta(self, Y, delta_next):
        return np.where(Y > 0, delta_next, 0)
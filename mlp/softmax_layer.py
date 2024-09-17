import numpy as np

class SoftmaxLayer(object):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        exp_X = np.exp(X - np.max(X))
        return exp_X / exp_X.sum(axis=1)[:, np.newaxis]

    def delta(self, Y, delta_next):
        return Y * (delta_next - (delta_next * Y).sum(axis=1)[:, np.newaxis])

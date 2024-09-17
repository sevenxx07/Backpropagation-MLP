import numpy as np

class LossCrossEntropy(object):
    def __init__(self, name):
        super(LossCrossEntropy, self).__init__()
        self.name = name

    def forward(self, X, T):
        """
        Forward message.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, 1)
        """
        return np.where(T == 1, -np.log(X), 0).sum(axis=1)

    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: delta vector from the loss layer, shape (n_samples, n_inputs)
        """
        return np.where(T == 1, -np.reciprocal(X), 0)


class LossCrossEntropyForSoftmaxLogits(object):
    def __init__(self, name):
        super(LossCrossEntropyForSoftmaxLogits, self).__init__()
        self.name = name

    def forward(self, X, T):
        return np.where(T == 1, -(X - np.max(X)), 0).sum(axis=1) + np.log(np.sum(np.exp(X - np.max(X)), axis=1))

    def delta(self, X, T):
        exp_X = np.exp(X - np.max(X))
        return exp_X / exp_X.sum(axis=1)[:, np.newaxis] - T

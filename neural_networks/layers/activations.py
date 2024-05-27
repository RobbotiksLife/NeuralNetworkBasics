import numpy as np
from neural_networks.layers.activation import Activation


class SoftPlus(Activation):
    def __init__(self):
        """
        Initialize the SoftPlus activation layer.

        The SoftPlus function is defined as log(1 + exp(x)).
        The derivative (gradient) of the SoftPlus function is exp(x) / (1 + exp(x)).
        The second derivative of the SoftPlus function is exp(x) / (1 + exp(x))^2.
        """
        super().__init__(self.softplus, self.softplus_prime)

    def softplus(self, x):
        return np.log(1 + np.exp(x))

    def softplus_prime(self, x):
        exp_x = np.exp(x)
        return exp_x / (1 + exp_x)



class Sigmoid(Activation):
    def __init__(self):
        """
        Initialize the SoftPlus activation layer.

        The SoftPlus function is defined as log(1 + exp(x)).
        The derivative (gradient) of the SoftPlus function is exp(x) / (1 + exp(x)).
        The second derivative of the SoftPlus function is exp(x) / (1 + exp(x))^2.
        """
        super().__init__(self.sigmoid, self.sigmoid_prime)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

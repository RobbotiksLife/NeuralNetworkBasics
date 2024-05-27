import numpy as np


class Layer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        pass

    def backward(self, y_gradient: np.array, learning_rate: float):
        pass
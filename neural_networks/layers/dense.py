import numpy as np
from neural_networks.layers.layer import Layer


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        # Initialize weights(output_size, input_size) and biases(output_size, 1)
        self.weights: np.array = np.random.randn(output_size, input_size)
        self.biases: np.array = np.random.rand(output_size, 1)

    def forward(self, x: np.array) -> np.array:
        """
            Perform the forward pass for the dense layer.

            Parameters:
            x (np.ndarray): Input matrix of shape (input_size, 1)

            Returns:
            np.ndarray: Output matrix of shape (output_size, 1)
        """
        self.x: np.array = x
        return np.dot(self.weights, self.x) + self.biases

    def backward(self, y_gradient: np.array, learning_rate: float) -> np.array:
        """
            Perform the backward pass for the dense layer, updating weights and biases.

            Parameters:
            y_gradient (np.ndarray): Gradient of the loss with respect to the output of this layer, of shape (output_size, 1)
            learning_rate (float): Learning rate for the gradient descent update step

            Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer, to be used in backpropagation, of shape (input_size, 1)
        """
        weights_gradient = np.dot(y_gradient, self.x.T)
        self.weights -= weights_gradient * learning_rate
        self.biases -= y_gradient * learning_rate
        return np.dot(self.weights.T, y_gradient)
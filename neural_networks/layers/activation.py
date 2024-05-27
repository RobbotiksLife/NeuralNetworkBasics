import numpy as np
from neural_networks.layers.layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        """
            Initialize the activation layer with the activation function and its derivative.

            Parameters:
            activation (callable): The activation function (e.g., np.tanh, sigmoid)
            activation_prime (callable): The derivative of the activation function
        """
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, x: np.array) -> np.array:
        """
            Perform the forward pass through the activation layer.

            Parameters:
            x (np.ndarray): Input matrix from the previous layer, of shape (input_size, 1)

            Returns:
            np.ndarray: Output matrix after applying the activation function, of shape (input_size, 1)
        """
        self.x = x
        return self.activation(x)

    def backward(self, y_gradient: np.array, learning_rate: float):
        """
            Perform the backward pass through the activation layer.

            Parameters:
            y_gradient (np.ndarray): Gradient of the loss with respect to the output of this layer, of shape (output_size, 1)
            learning_rate (float): Learning rate (not used in activation layer but kept for consistency)

            Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer, of shape (input_size, 1)
        """
        # Element-wise multiplication of the output gradient with the derivative of the activation function
        return np.multiply(y_gradient, self.activation_prime(self.x))
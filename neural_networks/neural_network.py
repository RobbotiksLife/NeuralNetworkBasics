from neural_networks.layers.layer import Layer
from neural_networks.loss_functions.loss_function import LossFunction
from loading_bar import LoadingBar
from typing import List
import numpy as np
import pickle


class NeuralNetwork:
    def __init__(self, layers: List[Layer], loss_function: LossFunction):
        self.layers = layers
        self.loss_function = loss_function

    def predict(self, x: np.array) -> np.array:
        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def update_params_with_backpropagation(self, y_true: np.array, y_pred: np.array):
        grad = self.loss_function.count_prime(y_true=y_true, y_pred=y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(y_gradient=grad, learning_rate=0.01)

    def train(self, x_train: np.array, y_train: np.array, epochs: int):
        loading_bar = LoadingBar(total=epochs)
        e = 1
        while e <= epochs:
            error = 0
            for x, y in zip(x_train, y_train):
                y_pred = self.predict(x)

                error += self.loss_function.count(y_true=y, y_pred=y_pred)

                self.update_params_with_backpropagation(y_true=y, y_pred=y_pred)
            error /= len(x_train)
            loading_bar.update(e, f"Epoch {e}/{epochs}, error={error:.4f}")
            e += 1
        loading_bar.finish()

    def save(self, filename):
        """
        Save the NeuralNetwork object to a file using pickle.

        Parameters:
        filename (str): The name of the file to which the object will be saved.
        """
        with open(f"{filename}.pkl", 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        """
        Load a NeuralNetwork object from a file using pickle.

        Parameters:
        filename (str): The name of the file from which the object will be loaded.

        Returns:
        NeuralNetwork: The loaded NeuralNetwork object.
        """
        with open(f"{filename}.pkl", 'rb') as file:
            neural_network = pickle.load(file)
        return neural_network

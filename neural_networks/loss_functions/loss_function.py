import numpy as np

class LossFunction:
    def __init__(self, loss, loss_prime):
        self._loss = loss
        self._loss_prime = loss_prime

    def count(self, y_true: np.array, y_pred: np.array) -> float:
        """
           Compute the loss between the true values and the predicted values.

           Parameters:
           y_true (np.ndarray): True values, with shape (some_size, 1)
           y_pred (np.ndarray): Predicted values, with shape (some_size, 1)

           Returns:
           float: loss function result
       """
        return self._loss(y_true, y_pred)

    def count_prime(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
            Compute the derivative (gradient) of the loss function with respect to the predicted values.

            Parameters:
            y_true (np.ndarray): True values, with shape (some_size, 1)
            y_pred (np.ndarray): Predicted values, with shape (some_size, 1)

            Returns:
            np.ndarray: The gradient of the loss function with respect to y_pred, with shape (some_size, 1)
        """
        return self._loss_prime(y_true, y_pred)
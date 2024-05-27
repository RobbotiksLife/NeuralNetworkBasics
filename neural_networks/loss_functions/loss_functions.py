from neural_networks.loss_functions.loss_function import LossFunction
import numpy as np


# class MeanSquaredErrorLoss(LossFunction):
#     def __init__(self):
#         super().__init__(
#             loss=lambda y_true, y_pred: np.mean(np.power(y_true - y_pred, 2)),
#             loss_prime=lambda y_true, y_pred: -2 * (y_true - y_pred) / np.size(y_true)
#         )

class MeanSquaredErrorLoss(LossFunction):
    def __init__(self):
        super().__init__(
            loss=self.mse_loss,
            loss_prime=self.mse_loss_prime
        )

    def mse_loss(self, y_true, y_pred):
        """
        Compute the Mean Squared Error (MSE) between the true values and the predicted values.

        Parameters:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

        Returns:
        float: The mean squared error
        """
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_loss_prime(self, y_true, y_pred):
        """
        Compute the derivative (gradient) of the Mean Squared Error (MSE) with respect to the predicted values.

        Parameters:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values

        Returns:
        np.ndarray: The gradient of the MSE with respect to y_pred
        """
        return -2 * (y_true - y_pred) / np.size(y_true)
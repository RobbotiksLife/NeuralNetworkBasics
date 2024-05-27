import numpy as np
from neural_networks.neural_network import NeuralNetwork
from neural_networks.loss_functions.loss_functions import MeanSquaredErrorLoss
from neural_networks.layers.dense import Dense
from neural_networks.layers.activations import SoftPlus


if __name__ == '__main__':
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

    neural_network = NeuralNetwork(
        layers=[
            Dense(2, 3),
            SoftPlus(),
            Dense(3, 1),
            SoftPlus()
        ],
        loss_function=MeanSquaredErrorLoss()
    )

    neural_network.train(X, y, epochs=10000)

    neural_network.save("neural_network_xor")
    neural_network = NeuralNetwork.load("neural_network_xor")

    print(neural_network.predict(X[0]))
    print(neural_network.predict(X[1]))
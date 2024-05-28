# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
#
# # Generate synthetic data
# np.random.seed(42)
# x1 = np.random.rand(1000) * 2 - 1  # Generate 1000 points between -1 and 1
# x2 = np.random.rand(1000) * 2 - 1
#
# # Labels: 1 if above y=x line, 0 if below
# labels = (x2 > x1).astype(int)
#
# # Combine x1 and x2 into a single dataset
# data = np.vstack((x1, x2)).T
#
# # Build the model
# model = Sequential([
#     Dense(2, input_dim=2, activation='relu'),  # Hidden layer with 2 neurons and ReLU activation
#     Dense(1, activation='sigmoid')  # Output layer with 1 neuron and sigmoid activation
# ])
#
# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Train the model
# model.fit(data, labels, epochs=50, batch_size=10, verbose=1)
#
# # Evaluate the model
# loss, accuracy = model.evaluate(data, labels)
# print(f"Loss: {loss}, Accuracy: {accuracy}")
#
# # Visualize the data
# plt.figure(figsize=(10, 6))
# plt.scatter(x1, x2, c=labels, cmap='bwr', alpha=0.7)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Data and Decision Boundary')
#
# # Plot decision boundary
# xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
# grid = np.c_[xx.ravel(), yy.ravel()]
# predictions = model.predict(grid).reshape(xx.shape)
#
# plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
# plt.colorbar()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt

import utils


def plot_data(dataset, class_column, filename, plot_columns):
    """
    Plot the data from the given dataset.

    Parameters:
        dataset (DataFrame): The dataset containing the data.
        class_column (str): The name of the column defining the class.
        filename (str): The name of the file to save the plot.
        plot_columns (list): List of columns to be plotted.
    """

    # Define the columns to keep
    keep_columns = plot_columns + [class_column]
    dataset = dataset[keep_columns]

    # Define a color map
    unique_classes = dataset[class_column].unique()
    colors = {cls: plt.cm.tab10(i) for i, cls in enumerate(unique_classes)}

    # Plot the data
    plt.figure(figsize=(10, 6))
    for class_name, color in colors.items():
        subset = dataset[dataset[class_column] == class_name]
        plt.scatter(subset[plot_columns[0]], subset[plot_columns[1]], color=color, label=class_name, edgecolors='w',
                    s=100)

    plt.title('Data Plot')
    plt.xlabel(plot_columns[0])
    plt.ylabel(plot_columns[1])
    plt.legend(title=class_column)
    plt.grid(True)
    plt.savefig(filename)

from utils import float_range
def plot_decision_regions(X, y, predict_func, filename="plot_name"):
    # Define the color map
    colors = ['red', 'green', 'blue']
    class_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    # Plot the data points
    for idx, class_label in enumerate(class_labels):
        plt.scatter(X[y[:, idx, 0] == 1, 0, 0], X[y[:, idx, 0] == 1, 1, 0], c=colors[idx], label=class_label)

    # Create a mesh grid
    x_min, x_max = X[:, 0, 0].min() - 1, X[:, 0, 0].max() + 1
    y_min, y_max = X[:, 1, 0].min() - 1, X[:, 1, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_reshaped = grid_points.reshape(-1, 2, 1)

    # Predict the class for each point in the mesh grid
    Z = np.array([predict_func(point) for point in grid_points_reshaped])
    Z = np.argmax(Z, axis=1)  # Get the index of the max class probability
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary by assigning a color to each point in the mesh grid
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    # plt.scatter(xx, yy, Z, alpha=0.05, cmap=plt.cm.RdYlBu)
    # for x in float_range(x_min, x_max, 0.1):
    #     for y in float_range(y_min, y_max, 0.1):
    #         plt.scatter(x, y, alpha=0.05, c=predict_func(np.array([[x], [y]])))

    # Add labels, legend, and show plot
    plt.xlabel('Sepal Width')
    plt.ylabel('Petal Width')
    plt.legend()
    plt.savefig(f"{filename}.png")

def to_one_hot(y_pred):
    y_pred = np.squeeze(y_pred)
    max_index = np.argmax(y_pred)
    one_hot = np.zeros_like(y_pred)
    one_hot[max_index] = 1
    return one_hot



import numpy as np
from neural_networks.neural_network import NeuralNetwork
from neural_networks.loss_functions.loss_functions import MeanSquaredErrorLoss
from neural_networks.layers.dense import Dense
from neural_networks.layers.activations import SoftPlus, Sigmoid, ReLU


if __name__ == '__main__':
    # Load the dataset
    iris_data = pd.read_csv(
        'iris/iris.data',
        header=None,
        names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    )

    # # Plot the data using the plot_data function
    # plot_data(
    #     iris_data,
    #     'class',
    #     'iris_plot_sepal_petal_width.png',
    #     plot_columns=['sepal_width', 'petal_width']
    # )

    # Drop columns
    X = iris_data[['sepal_width', 'petal_width']].to_numpy().reshape(150, 2, 1)
    y = np.reshape(iris_data['class'].map({
        'Iris-setosa': [1, 0, 0],
        'Iris-versicolor': [0, 1, 0],
        'Iris-virginica': [0, 0, 1]
    }).to_list(), (150, 3, 1))

    # neural_network = NeuralNetwork(
    #     layers=[
    #         Dense(2, 2),
    #         ReLU(),
    #         Dense(2, 3),
    #         Sigmoid()
    #     ],
    #     loss_function=MeanSquaredErrorLoss()
    # )
    #
    # neural_network.train(
    #     x_train=X,
    #     y_train=y,
    #     epochs=100000
    # )
    #
    # neural_network.save("neural_network_iris_flower_dataset_v6")
    neural_network = NeuralNetwork.load("neural_network_iris_flower_dataset_v6")

    print(to_one_hot(neural_network.predict(X[0])))
    print(to_one_hot(neural_network.predict(X[60])))
    print(to_one_hot(neural_network.predict(X[140])))

    print(X.shape)
    print(y.shape)

    # Define Accuracy
    y_pred_list = []
    correct_values_num = 0
    for x, y_true in zip(X, y):
        y_pred = to_one_hot(neural_network.predict(x))
        y_pred_list.append(y_pred)
        if np.array_equal(y_true.reshape(3), y_pred):
            correct_values_num += 1
        # else:
        #     print(x)
    print(f"Accuracy: {correct_values_num/len(y)}")


    y_pred_np_array = np.array(y_pred_list).reshape(150, 3, 1)
    plot_decision_regions(
        X=X,
        y=y_pred_np_array,
        predict_func=lambda x: to_one_hot(neural_network.predict(x)),
        filename="nn_prediction_results_v6"
    )



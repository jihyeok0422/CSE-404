import numpy as np  # functions, vectors, matrices, linear algebra...
from random import choice  # randomly select item from list
import random
import matplotlib.pyplot as plt  # plots


def generate_datapoints(size, limit):
    """
    This function generates random data points from -limit to limit
    :param size: the size of the data. How many points it is generating
    :param limit: the limit of the points
    :return: numpy matrix of data points
    """
    x_data = np.zeros(shape=(size, 3))
    for i in range(size):
        x1 = random.uniform(-limit, limit)
        x2 = random.uniform(-limit, limit)
        x_data[i] = [x1, x2, 1]
    return x_data


def generate_function(limit):
    """
    This function generates random function to divide the data points
    :param limit: limit of each weights can be
    :return: the numpy array of weights, w2 = bias
    """
    w0 = random.uniform(-limit, limit)
    w1 = random.uniform(-limit, limit)
    w2 = random.uniform(-limit, limit)
    return np.array([w0, w1, w2])


def create_labels(size, function, data):
    """
    This function creates labels of each data point, whether it is +1 or -1
    :param size: size of the data set
    :param function: Randomly generated slope to divide the dataset
    :param data: dataset with randomly generated data points
    :return:the array with labels, where each indices corresponds to each data point
    """
    y_data = []
    # eq = -function[1]/function[2] * x + -function[0]/function[2]
    for i in range(size):
        if function[1] == 0:
            num = 0
        else:
            num = -function[0] / function[1] * data[i][0] + -function[2] / function[1]
        if data[i][1] > num:
            y_data.append(1)
        else:
            y_data.append(-1)
    return y_data

def train_perceptron(training_data):
    '''
    Train a perceptron model given a set of training data
    :param training_data: A list of data points, where training_data[0]
    contains the data points and training_data[1] contains the labels.
    Labels are +1/-1.
    :return: learned model vector
    '''
    X = training_data[0]
    y = training_data[1]
    model_size = X.shape[1]
    w = np.zeros(model_size)    #np.random.rand(model_size)
    iteration = 1
    while True:
        # compute results according to the hypothesis
        results = np.sign(np.multiply(np.matmul(X, w), y))
        # get incorrect predictions (you can get the indices)
        indices = np.arange(X.shape[0])
        misclassified_indices = indices[results != 1]
        # Check the convergence criteria (if there are no misclassified
        # points, the PLA is converged and we can stop.)
        if len(misclassified_indices) == 0:
            break

        # Pick one misclassified example.
        pick = choice(misclassified_indices)
        x_star, y_star = X[pick], y[pick]
        # Update the weight vector with perceptron update rule
        w += y_star * x_star
        iteration += 1
    plt.title("total iterations: " + format(iteration))
    print("{}: total iterations", format(iteration))
    return w


def make_graph(limit, function, size, data, perceptron, labels):
    """
    This function draw a graph with data, function, and perceptron
    :param limit: the limit of the data points
    :param function: randomly generated weights
    :param size: size of the dataset
    :param data: randomly generated dataset
    :param perceptron: the trained perceptron
    :param labels: array that indicate whether each point is under or above the function
    :return: none
    """
    for i in range(size):
        if labels[i] == 1:
            plt.plot(data[i][0], data[i][1], 'or')
        else:
            plt.plot(data[i][0], data[i][1], 'ob')
    plot_slope(function, 'm', 'Target f(x)')
    plot_slope(perceptron, 'y', 'Perceptron g(x)')

    plt.xlim([-limit - 1, limit + 1])
    plt.ylim([-limit - 1, limit + 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def plot_slope(function, color, name):
    """
    This function draws the slope using the function and limit
    :param function: the weights to draw the slope
    :param limit: limit of the data points
    :param color: color of the slope
    :param name: name of the slope
    :return: none
    """
    if function[1] == 0:
        slope = 0
        intersection = 0
    else:
        slope = -function[0]/function[1]
        intersection = -function[2]/function[1]
    plt.plot(np.arange(-100, 100), slope*np.arange(-100, 100) + intersection, color, label = name)
    plt.legend()

if __name__ == '__main__':


    # rnd_x = np.array([[0,1,1],\
    #                   [0.6,0.6,1],\
    #                   [1,0,1],\
    #                   [1,1,1],\
    #                   [0.3,0.4,1],\
    #                   [0.2,0.3,1],\
    #                   [0.1,0.4,1],\
    #                   [0.5,-0.1,1]])
    #
    # rnd_y = np.array([1,1,1,1,-1,-1,-1,-1])
    # rnd_data = [rnd_x,rnd_y]
    # trained_model = train_perceptron(rnd_data)
    # print("Model:", trained_model)
    # print_prediction(trained_model, rnd_x)
    # make_graph(1, 8, rnd_x, trained_model, rnd_y)

    data_limit = 10
    size = 1000
    function_limit = 20
    data = generate_datapoints(size, data_limit)
    function = generate_function(function_limit)
    labels = create_labels(size, function, data)
    training_data = [data, labels]
    perceptron = train_perceptron(training_data)
    make_graph(data_limit, function, size, data, perceptron, labels)
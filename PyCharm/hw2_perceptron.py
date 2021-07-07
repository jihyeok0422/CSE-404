import numpy as np  #functions, vectors, matrices, linear algebra...
from random import choice   #randomly select item from list
import matplotlib.pyplot as plt #plots

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
    print("{}: total iterations", format(iteration))
    return w

def print_prediction(model,data):
    '''
    Print the predictions given the dataset and the learned model.
    :param model: model vector
    :param data:  data points
    :return: nothing
    '''
    result = np.matmul(data,model)
    predictions = np.sign(result)
    for i in range(len(data)):
        print("{}: {} -> {}".format(data[i][:2], result[i], predictions[i]))

def make_graph(limit, size, data, perceptron, lables):
    for i in range(size):
        if lables[i] == 1:
            plt.plot(data[i][0], data[i][1], 'or')
        else:
            plt.plot(data[i][0], data[i][1], 'ob')
    plot_slope(perceptron, limit)

    plt.xlim([-limit - 1, limit + 1])
    plt.ylim([-limit - 1, limit + 1])
    plt.show()

def plot_slope(function, limit):
    if function[1] == 0:
        slope = 0
        intersection = 0
    else:
        slope = -function[0]/function[1]
        intersection = -function[2]/function[1]
    plt.plot(np.arange(-limit-1, limit+1), slope*np.arange(-limit-1, limit+1) + intersection)



if __name__ == '__main__':

    rnd_x = np.array([[0,1,1],\
                      [0.6,0.6,1],\
                      [1,0,1],\
                      [1,1,1],\
                      [0.3,0.4,1],\
                      [0.2,0.3,1],\
                      [0.1,0.4,1],\
                      [0.5,-0.1,1]])

    rnd_y = np.array([1,1,1,1,-1,-1,-1,-1])
    rnd_data = [rnd_x,rnd_y]

    trained_model = train_perceptron(rnd_data)
    print("Model:", trained_model)
    print_prediction(trained_model, rnd_x)
    make_graph(1, 8, rnd_x, trained_model, rnd_y)





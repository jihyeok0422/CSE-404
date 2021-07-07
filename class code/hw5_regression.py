# CSE 404 Intro to Machine Learning
# Homework 5: Linear Regression


import time
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from random import randrange



def rand_split_train_test(data, label, train_perc):
    if train_perc >= 1 or train_perc <= 0:
        raise Exception('train_perc should be between (0,1).')
    sample_size = data.shape[0]
    if sample_size < 2:
        raise Exception('Sample size should be larger than 1. ')

    num_train_sample = np.max([np.floor(sample_size * train_perc).astype(int), 1])
    data, label = shuffle(data, label)

    data_tr = data[:num_train_sample]
    data_te = data[num_train_sample:]

    label_tr = label[:num_train_sample]
    label_te = label[num_train_sample:]

    return data_tr, data_te, label_tr, label_te



def subsample_data(data, label, subsample_size):
    # protected sample size
    subsample_size = np.max([1, np.min([data.shape[0], subsample_size])])
    data, label = shuffle(data, label)
    data = data[:subsample_size]
    label = label[:subsample_size]
    return data, label



def generate_rnd_data(feature_size, sample_size, bias=False):
    # Generate X matrix
    data = np.concatenate((np.random.randn(sample_size, feature_size), np.ones((sample_size, 1))), axis=1) \
        if bias else np.random.randn(sample_size, feature_size)  # the first dimension is sample_size (n X d)

    # Generate ground truth model
    truth_model = np.random.randn(feature_size + 1, 1) * 10 \
        if bias else np.random.randn(feature_size, 1) * 10

    # Generate label
    label = np.dot(data, truth_model)

    # Add element-wise Gaussian noise to each label
    label += np.random.randn(sample_size, 1)
    return data, label, truth_model



# Sine Function
def sine_data(sample_size, order_M, plot_data = False, noise_level = 0.1, bias = False):
    if int(order_M) != order_M: 
        raise Exception('order_M should be an integer.')
    if order_M < 0:
        raise Exception('order_M should be at least larger than 0.')
    
    # Generate X matrix
    x = np.random.rand(sample_size,1) * 2 * np.pi        # generate x from 0 to 2pi
    X = np.column_stack([ x**m for m in range(order_M)])

    data = np.concatenate((X, np.ones((sample_size, 1))), axis=1) if bias else X

    # Ground truth model: a sine function
    f = lambda x: np.sin(x)

    # Generate labels
    label = f(x)

    # Add element-wise Gaussian noise to each label
    label += np.random.randn(sample_size, 1)*noise_level

    if plot_data:
        plt.figure()
        xx = np.arange(0, np.pi * 2, 0.001)
        yy = f(xx)
        plt.plot(xx, yy, linestyle = '-', color = 'g', label = 'Objective Value')
        plt.scatter(x, label, color = 'b', marker = 'o', alpha = 0.3)
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title("Sine Data (N = %d) with Noise Level %.4g.".format(sample_size, noise_level))
        plt.show()

    return data, label, f


######################################################################################

def mean_squared_error(true_label, predicted_label):
    """
        Compute the mean square error between the true and predicted labels
        :param true_label: Nx1 vector
        :param predicted_label: Nx1 vector
        :return: scalar MSE value
    """
    #TODO
    N = len(true_label)
    sum = 0
    for i in range(N):
        difference = true_label[i] - predicted_label[i]
        square = difference ** 2
        sum += square

    mse = sum / N
    mse = np.sqrt(mse)
    return mse



def least_squares(feature, target):
    """
        Compute the model vector obtained after MLE
        w_star = (X^T X)^(-1)X^T t
        :param feature: Nx(d+1) matrix
        :param target: Nx1 vector
        :return: w_star (d+1)x1 model vector
        """
    #TODO
    w_star = np.dot(np.linalg.inv(np.dot(feature.T, feature)), np.dot(feature.T, target))
    return w_star



def ridge_regression(feature, target, lam = 1e-17):
    """
        Compute the model vector when we use l2-norm regularization
        x_star = (X^T X + lambda I)^(-1) X^T t
        :param feature: Nx(d+1) matrix
        :param target: Nx1 vector
        :param lam: the scalar regularization parameter, lambda
        :return: w_star (d+1)x1 model vector
        """
    #TODO
    column_num = feature.shape[1]
    XtX = np.dot(feature.T, feature)
    XtX_lam = XtX + lam*np.identity(column_num)
    inv = np.linalg.inv(XtX_lam)
    w_star = np.dot(inv, np.dot(feature.T, target))
    return w_star



# K-fold cross validation
def k_fold(current_fold, total_fold, data_split, label_split):
    #TODO
    if current_fold >= total_fold:
        raise Exception('current_fold out of range.')
    data_tr = list(data_split)
    label_tr = list(label_split)
    data_te = data_tr.pop(current_fold)
    label_te = label_tr.pop(current_fold)
    data_train = np.zeros(shape = (len(data_tr[0]) * len(data_tr), len(data_tr[0][0])))
    label_train = np.zeros(shape = (len(label_tr[0]) * len(label_tr), 1))
    i = 0
    for index in range(len(data_tr)):
        for row in range(len(data_tr[index])):
            data_train[i] = data_tr[index][row]
            i += 1
    i = 0
    for index in range(len(label_tr)):
        for row in range(len(label_tr[index])):
            label_train[i] = label_tr[index][row]
            i += 1
    return data_train, np.asarray(data_te), label_train, np.asarray(label_te)


########################################################################################

def compute_gradient(feature, target, weight, lam = 1e-17):
    # Compute the gradient of linear regression objective function with respect to w
    gradient = np.dot(feature.T, np.dot(feature, weight) - target) + lam*weight
    return gradient



def compute_objective_value(feature, target, model):
    # Compute MSE
    m = len(target)
    prediction = np.dot(feature, model)
    mse = (1/(2 * m)) * np.sum(np.square(prediction - target))
    return mse


# Gradient Descent
def gradient_descent(feature, target, step_size, max_iter, lam = 1e-17):
    weight = np.random.randn(len(feature[0]), 1)
    obj_vals = np.zeros(max_iter)
    for i in range(max_iter):
        # Compute gradient
        gradient = compute_gradient(feature, target, weight, lam)
        # Update the model
        weight = weight - step_size * gradient
        # Compute the error (objective value)
        obj_vals[i] = compute_objective_value(feature, target, weight)
    return weight, obj_vals


# Stochastic Gradient Descent
def batch_gradient_descent(feature, target, step_size, max_iter, batch_size, lam = 1e-17):
    feature_train, target_train = subsample_data(feature, target, batch_size)
    weight = np.random.randn(len(feature_train[0]), 1)
    obj_vals = np.zeros(max_iter)
    for i in range(max_iter):
        # Compute gradient
        gradient = compute_gradient(feature_train, target_train, weight, lam)
        # Update the model
        weight = weight - step_size * gradient
        # Compute the error (objective value)
        obj_vals[i] = compute_objective_value(feature_train, target_train, weight)
    return weight, obj_vals

# Plots/Errors
# def plot_objective_function(objective_value, batch_size=None):
# def print_train_test_error(train_data, test_data, train_label, test_label, model):

##########################################################################################

# TODO: Homework Template
if __name__ == '__main__':
    plt.interactive(False)

    np.random.seed(491)


    # Problem 1
    # Complete Least Squares, Ridge Regression, MSE
    # Randomly generate & plot 30 data points using sine function
    data, label, f = sine_data(30, 10, True, 0.3, True)
    # Randomly split the dataset
    data_tr, data_te, label_tr, label_te = rand_split_train_test(data, label, 0.70)
    # For each lambda, use Ridge Regression to calculate & plot MSE for training & testing sets
    lambda_array = [10**-10, 10**-5, 10**-2, 10**-1, 1, 10, 100, 1000]
    train_performance = []
    test_performance = []
    for lamb in lambda_array:
        # w_star is like a theta_hat
        w_star = ridge_regression(data_tr, label_tr, lamb)
        train_predict = np.dot(data_tr, w_star)
        test_predict = np.dot(data_te, w_star)
        train_performance += [mean_squared_error(label_tr, train_predict)]
        test_performance += [mean_squared_error(label_te, test_predict)]
    plt.plot(lambda_array, train_performance, "b-", label = 'Training Error')
    plt.plot(lambda_array, test_performance, "r-", label = 'Testing Error')
    plt.xlabel("lambda")
    plt.ylabel("Mean Squared Error")
    plt.xlim((0, 1000))
    plt.legend(loc = "upper right")
    plt.title("Ridge Regression Performance")
    plt.show()

	# Implement k-fold CV & choose best lambda
    mse_list = []
    folds = 4
    for lamb2 in lambda_array:
        data_split = list()
        data_copy = list(data)
        fold_size = int(len(data) / folds)
        label_split = list()
        label_copy = list(label)
        # dividing the data into 4 (depends on folds) pieces
        for i in range(folds):
            fold_data = list()
            fold_label = list()
            while len(fold_data) < fold_size:
                index = randrange(len(data_copy))
                fold_data.append(data_copy.pop(index))
                fold_label.append(label_copy.pop(index))
            data_split.append(fold_data)
            label_split.append(fold_label)
        sum_mse = 0
        # calculating average mse of this lambda using the data from cross-validation
        for fold in range(folds):
            data_train, data_test, label_train, label_test = k_fold(fold, folds, data_split, label_split)
            w_star = ridge_regression(data_train, label_train, lamb2)
            prediction_te = np.dot(data_test, w_star)
            sum_mse += mean_squared_error(label_test, prediction_te)
        avg_mse = sum_mse / folds
        mse_list.append(avg_mse)
    min_mse = min(mse_list)
    index = mse_list.index(min_mse)
    print("minimum MSE: ", min_mse)
    print("Best lambda to choose: ", lambda_array[index])

    # Problem 2
    # Complete Gradient Descent & Stochastic GD
    # Implement ridge regression with GD & plot objectives at each iteration
    data, label, truth_model = generate_rnd_data(50, 1000, True)
    weight, obj_vals = gradient_descent(data, label, 0.001, 1000)
    # print(weight)
    # print(obj_vals)
    plt.plot(obj_vals)
    plt.xlabel("Iterations")
    plt.ylabel("Objective Value")
    plt.xlim([0, 10])
    plt.ylim([0, 100])
    plt.title("Gradient Descent: Objective per iteration")
    plt.show()
    # Implement SGD & plot objectives at each iteration per batch
    batch_array = [5, 10, 100, 500]
    for batch in batch_array:
        weight_batch, obj_vals_batch = batch_gradient_descent(data, label, 0.001, 1000, batch)
        plt.plot(obj_vals_batch, label = "batch size :%i" %batch)
        plt.xlabel("Iterations")
        plt.ylabel("Objective Value")
        plt.xlim([0, 100])
        plt.ylim([0, 1000])
    plt.legend(loc = "upper right")
    plt.title("Batch Gradient Descent: Objective per iteration")
    plt.show()
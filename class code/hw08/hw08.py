from cvxopt import matrix, solvers
import numpy as np
import random

def QPSolver(X, y):
    m, n = X.shape
    Q = np.identity(n)
    temp_row = np.zeros((1, n))
    temp_col = np.zeros((n+1, 1))
    Q = np.vstack((temp_row, Q))
    Q = np.column_stack((temp_col, Q))

    p = np.zeros((1, n+1)).T
    A = np.zeros((m, n+1))
    for i in range(m):
        for j in range(n+1):
            if j == 0:
                A[i][j] = y.item(i)
            else:
                A[i][j] = X[i, j-1] * y.item(i)
    c = -np.ones((1, m)).T

    Q = matrix(Q) * 1.0
    p = matrix(p) * 1.0
    A = matrix(A) * -1.0
    c = matrix(c) * 1.0

    solve = solvers.qp(Q, p, A, c)
    return solve

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

if __name__ == '__main__':
    # print("test")
    # Q = matrix([[0.0, 0.0, 0.0],
    #             [0.0, 1.0, 0.0],
    #             [0.0, 0.0, 1.0]])
    # # print(Q)
    # p = matrix([0.0, 0.0, 0.0])
    # # print(p)
    # A = matrix([[-1.0, 0.0, 0.0],
    #             [-1.0, -2.0, -2.0],
    #             [1.0, 2.0, 0.0],
    #             [1.0, 3.0, 0.0]]).T
    # # print(A)
    # c = matrix([-1.0, -1.0, -1.0, -1.0])
    # print(c)
    # solve = solvers.qp(Q, p, A, c)
    # print(solve['x'].T)
    #
    # X = np.matrix([[0, 2, 2, 3],
    #                [0, 2, 0, 0]]).T
    # y = np.matrix([[-1, -1, 1, 1, 1, -1, 1, -1, 1]]).T
    #
    sample_size = 10
    dimension = 5000
    X = np.random.randint(10, size=(dimension, sample_size)).T
    # print(X)
    temp_function = generate_function(5)
    # print(X)
    y = np.matrix(create_labels(sample_size, temp_function, X))
    # print(y)
    # y = np.random.randint(2, size=(1, 5)).T
    # for i in range(5):
    #     if y[i, 0] == 0:
    #         y[i][0] = -1
    # try:
    temp = QPSolver(X, y)
    print(temp['x'])
    # except Exception:
    #     print("Linearly not separable")

    # plt.scatter()

# CSE 404 Intro to Machine Learning
# Homework 5: Linear Regression


import time
import numpy as np
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from random import randrange

def intensity_symmetry(filename, third_order = False):
    train_file = open(filename, "r")
    num_one = 0
    num_five = 0
    intensity_one = []
    intensity_five = []
    symmetry_one = []
    symmetry_five = []
    target = []
    data_matrix = np.empty((0, 2), float)
    if third_order:
        data_matrix = np.empty((0, 9), float)
    for i in train_file:
        fields = i.split()
        if float(fields[0]) == 1 or float(fields[0]) == 5:
            list = np.array(fields[1:257])
            float_matrix = [float(string_num) for string_num in list]
            float_matrix = np.reshape(float_matrix, (16, 16))
            intense = np.sum(float_matrix) / 256

            # splitting the matrix and flip it to check symmetry
            split_matrix = np.split(float_matrix, 2)
            lower = split_matrix[1]
            upper = np.flipud(split_matrix[0])
            sym = np.sum(np.absolute(np.subtract(upper, lower))) / 256.0
            if third_order:
                x = intense
                y = sym
                third = np.array([[x, y, x**2, x*y, y**2, x**3, (x**2)*y, x*(y**2), y**3]])
                data_matrix = np.append(data_matrix, third, axis=0)
            else:
                data_matrix = np.append(data_matrix, np.array([[intense, sym]]), axis=0)
            if float(fields[0]) == 1:
                if num_one < 2:
                    num_one += 1
                    # Part A
                    plt.imshow(float_matrix, cmap="gray")
                    plt.show()
                intensity_one.append(intense)
                # Calculate the difference of the two matrices
                symmetry_one.append(sym)
                target.append(1)
            elif float(fields[0]) == 5:
                if num_five < 2:
                    num_five += 1
                    # part A
                    plt.imshow(float_matrix, cmap="gray")
                    plt.show()
                intensity_five.append(intense)
                # Calculate the difference of the two matrices
                symmetry_five.append(sym)
                target.append(-1)
    train_file.close()

    return intensity_one, intensity_five, symmetry_one, symmetry_five, data_matrix, target

def compute_E(feature, target, weight):
    # 1/N * sum(ln(1 + exp(-y * wTx)))
    E = 0
    sum = 0
    for i in range(len(feature)):
        e_sum = 0
        for j in range(len(weight[0])):
            e_sum += weight[0][j]*feature[i][j]
        sum = np.log(1 + np.exp(target[i] * e_sum))
    E = sum / len(feature)
    return E

def perceptron(feature, target, max_iter):
    weight = np.random.randn(1, len(feature[0]))
    for i in range(max_iter):
        for j in range(len(feature)):
            if np.dot(weight, feature[j]) * target[j] <= 0:
                # weight[0][0] += feature[j][0] * target[j]
                # weight[0][1] += feature[j][1] * target[j]
                # weight[0][2] += feature[j][2] * target[j]
                for z in range(len(weight[0])):
                    weight[0][z] += feature[j][z] * target[j]
    return weight

# TODO: Homework Template
if __name__ == '__main__':
    plt.interactive(False)

    np.random.seed(491)

    # Question 3
    intensity_one, intensity_five, symmetry_one, symmetry_five, data_matrix, target = intensity_symmetry("ZipDigits.train")

    # Part B
    print("intensity of One: ", intensity_one)
    print("intensity of Five: ", intensity_five)
    print("Symmetry of One: ", symmetry_one)
    print("Symmetry of Five: ", symmetry_five)

    # Part C
    plt.scatter(intensity_one, symmetry_one, label="One", color="none", edgecolors="blue")
    plt.scatter(intensity_five, symmetry_five, label="Five", color="red", marker="x")
    plt.xlabel("Intensity")
    plt.ylabel("Asymmetry")
    plt.title("Question 4 part A using Training")
    # Problem 4
    # part A
    data = np.column_stack((data_matrix, np.ones((len(data_matrix), 1))))
    target = np.array(target)
    weight = perceptron(data, target, 1000)
    w0 = weight.item(0)
    w1 = weight.item(1)
    w2 = weight.item(2)
    slope = -w0 / w1
    intersection = -w2 / w1
    x_val = np.linspace(-1, 0.2)
    y_val = (intersection + np.dot(slope, x_val))
    plt.xlim(-1, 0.2)
    plt.ylim(-0.1, 0.6)
    plt.plot(x_val, y_val, color='green', label="Decision Boundary")
    plt.legend()
    plt.show()

    # E on my training data
    print("E_in: ", compute_E(data, target, weight))

    # using test data
    intensity_one, intensity_five, symmetry_one, symmetry_five, data_matrix, target = intensity_symmetry(
        "ZipDigits.test")
    plt.scatter(intensity_one, symmetry_one, label="One", color="none", edgecolors="blue")
    plt.scatter(intensity_five, symmetry_five, label="Five", color="red", marker="x")
    plt.xlabel("Intensity")
    plt.ylabel("Asymmetry")
    plt.title("Question 4 part A using Testing")

    data = np.column_stack((data_matrix, np.ones((len(data_matrix), 1))))
    target = np.array(target)
    weight = perceptron(data, target, 1000)
    w0 = weight.item(0)
    w1 = weight.item(1)
    w2 = weight.item(2)
    slope = -w0 / w1
    intersection = -w2 / w1
    x_val = np.linspace(-1, 0.2)
    y_val = (intersection + np.dot(slope, x_val))
    plt.xlim(-1, 0.2)
    plt.ylim(-0.1, 0.6)
    plt.plot(x_val, y_val, color='green', label="Decision Boundary")
    plt.legend()
    plt.show()

    # E on my testing data
    print("E_test: ", compute_E(data, target, weight))

    # Third Order Training data
    intensity_one, intensity_five, symmetry_one, symmetry_five, data_matrix, target = intensity_symmetry(
        "ZipDigits.train", True)
    data = np.column_stack((data_matrix, np.ones((len(data_matrix), 1))))
    target = np.array(target)
    weight = perceptron(data, target, 1000)
    print("E_train_3rd: ", compute_E(data, target, weight))

    # Third Order Testing data
    intensity_one, intensity_five, symmetry_one, symmetry_five, data_matrix, target = intensity_symmetry(
        "ZipDigits.test", True)
    data = np.column_stack((data_matrix, np.ones((len(data_matrix), 1))))
    target = np.array(target)
    weight = perceptron(data, target, 1000)
    print("E_test_3rd: ", compute_E(data, target, weight))

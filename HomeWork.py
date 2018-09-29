# -*- MLP HomeWork -*-
# -*- @author: HW  -*-
# -*- encoding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# active function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # load data set
    data = np.loadtxt("Data1.txt")
    A_input = np.c_[data, np.ones([9, 1])]
    temp = np.zeros([9, 1])
    for i in range(5, 9):
        temp[i] = 1
    label = temp

    # setting MLP
    rate = 0.005
    iterations = 2000
    W = np.ones([3, 1])    # Weight

    # train
    for i in range(iterations):
        A_out = sigmoid(np.dot(A_input, W))
        A_delta = (A_out - label) * A_out * (1 - A_out)
        W = W - rate * np.dot(A_input.T, A_delta)
    print("============== Weight =========================")
    print(W)

    # Plotting Line
    plt.scatter(A_input[0:5, 0], A_input[0:5, 1], color='red', s=500, marker='+')
    plt.scatter(A_input[5:10, 0], A_input[5:10, 1], color='blue', s=500, marker='*')
    X = np.arange(-4, 6, 0.1)
    Y = (W[0] * X + W[2]) / (-1 * (W[1]))
    plt.plot(X, Y, label="line")
    plt.title("My Assignment")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()



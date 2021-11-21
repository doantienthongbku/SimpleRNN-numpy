import numpy as np
from utils import tanh, softmax, dtanh
import matplotlib.pyplot as plt
import copy


class RNN:
    def __init__(self):
        # Init parameters and derivative of them
        self.W_xh = np.random.rand(3, 4) * 0.01
        self.W_hh = np.random.rand(3, 3) * 0.01
        self.W_hy = np.random.rand(4, 3) * 0.01
        self.b_h = np.zeros(shape=(3, 1))
        self.b_y = np.zeros(shape=(4, 1))

        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_h = np.zeros_like(self.b_h)
        self.db_y = np.zeros_like(self.b_y)

        # Init state
        self.z = {}
        self.h = {}
        self.p = {}
        self.y_hat = {}

        self.dz = {}
        self.dh = {}
        self.dp = {}
        self.dy_hat = {}

        # save cost history
        self.cost_lst = []

    def forward(self, x):
        self.z[0] = self.W_xh @ x[0] + self.b_h
        self.h[0] = tanh(self.z[0])
        self.p[0] = self.W_hy @ self.h[0] + self.b_y
        self.y_hat[0] = softmax(self.p[0])

        for i in range(1, 4):
            self.z[i] = self.W_hh @ self.h[i - 1] + self.W_xh @ x[i] + self.b_h
            self.h[i] = tanh(self.z[i])
            self.p[i] = self.W_hy @ self.h[i] + self.b_y
            self.y_hat[i] = softmax(self.p[i])

        self.dz = copy.deepcopy(self.z)
        self.dh = copy.deepcopy(self.h)
        self.dp = copy.deepcopy(self.p)
        self.dy_hat = copy.deepcopy(self.y_hat)

    def cost_function(self, y):
        cross_entropy_cost = 0
        for i in range(4):
            cross_entropy_cost += - np.sum(y[i] * np.log(self.y_hat[i]) + (1 - y[i]) * np.log(1 - self.y_hat[i]))
        return cross_entropy_cost

    def fit(self, x, y, epochs, lr, print_cost=False):

        for epoch in range(epochs):
            # Forward
            self.forward(y)

            # Compute cost
            cost = self.cost_function(y)
            self.cost_lst.append(cost)

            # Print cost
            if print_cost and epoch % 100 == 0 or epoch == epochs - 1:
                print("Cost after iteration {}: {}".format(epoch, cost))

            # Reset derivative of parameters
            self.reset_epoch()

            # Backward
            for i in reversed(range(1, 4)):
                self.dy_hat[i] = - (np.divide(y[i], self.y_hat[i]) - np.divide(1 - y[i], 1 - self.y_hat[i]))
                # print(self.dy_hat[i])

                # print(dsoftmax(self.p[i]))
                # print(self.p[i].shape)
                self.dp[i] = self.dy_hat[i] * softmax(self.p[i], derivative=True)
                # print(self.dp[i] @ (self.h[i].T))
                self.dW_hy += self.dp[i] @ (self.h[i].T)
                self.db_y += np.sum(self.dp[i], axis=1, keepdims=True)

                self.dh[i] = self.W_hy.T @ (self.dp[i])
                self.dz[i] = self.dh[i] * dtanh(self.z[i])
                self.dW_hh += np.dot(self.dz[i], self.h[i - 1].T)
                self.dW_xh += np.dot(self.dz[i], x[i].T)
                self.db_h += np.sum(self.dz[i], axis=1, keepdims=True)

            # Backward for 0_th layer
            self.dy_hat[0] = - (np.divide(y[0], self.y_hat[0]) - np.divide(1 - y[0], 1 - self.y_hat[0]))

            self.dp[0] = self.dy_hat[0] * softmax(self.p[0], derivative=True)
            self.dW_hy += self.dp[0] @ (self.h[0].T)
            self.db_y += np.sum(self.dp[0], axis=1, keepdims=True)

            self.dh[0] = np.dot(self.W_hy.T, self.dp[0])
            self.dz[0] = self.dh[0] * dtanh(self.z[0])
            self.dW_xh += np.dot(self.dz[0], x[0].T)
            self.db_h += np.sum(self.dz[0], axis=1, keepdims=True)

            # Update parameters
            self.W_hy = self.W_hy - lr * self.dW_hy
            self.W_hh = self.W_hh - lr * self.dW_hh
            self.W_xh = self.W_xh - lr * self.dW_xh
            self.b_y = self.b_y - lr * self.db_y
            self.b_h = self.b_h - lr * self.db_h

    def reset_epoch(self):
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_h = np.zeros_like(self.b_h)
        self.db_y = np.zeros_like(self.b_y)

    def plot_costs(self, lr):
        plt.plot(self.cost_lst)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate = " + str(lr))
        plt.show()

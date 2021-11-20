import numpy as np
from utils import tanh, dtanh, softmax, dsoftmax
import matplotlib.pyplot as plt


class RNN:
    def __init__(self):

        self.W_xh = np.random.rand(3, 4) * 0.01
        self.W_hh = np.random.rand(3, 3) * 0.01
        self.W_hy = np.random.rand(4, 3) * 0.01
        self.b_h = np.zeros(shape=(1, 1))
        self.b_y = np.zeros(shape=(4, 1))
        self.z = {}
        self.h = {}
        self.p = {}
        self.y_hat = {}

        self.dp = np.zeros_like(self.p)
        self.dz = np.zeros_like(self.z)
        self.dy_hat = np.zeros_like(self.y_hat)
        self.dh = np.zeros_like(self.h)
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_h = np.zeros_like(self.b_h)
        self.db_y = np.zeros_like(self.b_y)

        self.cost_lst = []

    def forward(self, x):
        for i in range(4):
            self.z[i] = self.W_hh * self.h[i - 1] + self.W_xh * x[i] + self.b_h
            self.h[i] = tanh(self.z[0])
            self.p[i] = self.W_hy * self.h[i] + self.b_y
            self.y_hat[i] = softmax(self.p[i])

    def cost_function(self, y):
        cross_entropy_cost = - np.sum(y * np.log(self.y_hat) + (1 - y) * np.log(1 - self.y_hat))
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

            for i in reversed(range(1, 4)):
                # Backward
                self.dy_hat[i] = - (np.divide(y[i], self.y_hat[i]) - np.divide(1 - y[i], 1 - self.y_hat[i]))

                self.dp[i] = self.dy_hat[i] * dsoftmax(self.p[i])
                self.dW_hy = np.dot(self.dp[i], self.h[i].T)
                self.db_y = np.sum(self.dp[i], axis=1, keepdims=True)

                self.dh[i] = np.dot(self.W_hy[i].T, self.dp[i])
                self.dz[i] = self.dh[i] * dtanh(self.z[i])
                self.dW_hh = np.dot(self.dz[i], self.h[i - 1])
                self.dW_xh = np.dot(self.dz[i], self.x[i])
                self.db_h = np.sum(self.dz[i], axis=1, keepdims=True)

                # Update parameters
                self.W_hy = self.W_hy - lr * self.dW_hy
                self.W_hh = self.W_hh - lr * self.dW_hh
                self.W_xh = self.W_xh - lr * self.dW_xh
                self.b_y = self.b_y - lr * self.db_y
                self.b_h = self.b_h - lr * self.db_h

    def plot_costs(self, lr):
        plt.plot(self.cost_lst)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(lr))
        plt.show()

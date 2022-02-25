"""

draw line to data
measure distance from line (residuals)
    square distances and sum them up
rotate the line a little bit
    measure residuals then sum up squares
rotate line more
    measure residuals... etc

plot the sum squared of residuals and the corresponding rotation
    take the slope that has the least sum of squares  

variance(mean) = SS(mean)/n     where SS(mean) is the sum of distances squared to the mean of the y values
variance(fitted line) = SS(fit)/n
R^2 = (var(mean) - var(fit))/var(mean) OR (SS(mean) - SS(fit))/SS(mean)
    "there is a 60% reduction in variance when taking _ into account"

how to determine if R^2 is significant

F = a/b

a = (SS(mean) - SS(fit)) / (#paramsFit - #paramsMean)
b = SS(fit) / (n - #paramsFit)

where a = variation in mouse size explained by params
        b = variation in mouse size not explained by params

"""

from tkinter import NE
from black import Line
import pandas as pd
import numpy as np
import preprocessing

class LinearRegressionModel:
    def __init__(self, learningRate = .1, nEpochs = 1000):
        self.yIntercept = 0
        self.params = []
        self.learningRate = learningRate
        self.nEpochs = nEpochs
        self.coefficients = None

    def fit(self, x, y):
        mse, coefficients = self.gradient_descent(x, y, self.learningRate, self.nEpochs)

    def gradient_descent(self, x, y, L:float, nepochs):
        m = x.shape[0]
        ones = np.ones((m,1))
        x = np.concatenate((ones, x), axis = 1)
        n = x.shape[1]
        coefficients = np.ones(n)
        h = np.dot(x, coefficients)

        cost = np.ones(nepochs)
        for i in range(0, nepochs):
            coefficients[0] = coefficients[0] - (L / x.shape[0]) * sum(h-y)
            for j in range(1, n):
                # print((L/m) * sum(h-y) * x[:, j])
                coefficients[j] = coefficients[j] - (L / x.shape[0]) * sum((h-y) * x[:, j])
            h = np.dot(x, coefficients)
            cost[i] = 1/(2*m) * sum(np.square(h - y))

        print(cost)
        print(coefficients)
        return cost, coefficients

# I commented this out because I moved it to main.py

# lr = LinearRegressionModel(nEpochs=100)

# dt = preprocessing.importData()
# x = dt.drop(columns=['ETH']).to_numpy()[:-1]
# y = dt['ETH'].to_numpy()[1:]

# # print(x)
# # print(y)

# lr.fit(x, y)
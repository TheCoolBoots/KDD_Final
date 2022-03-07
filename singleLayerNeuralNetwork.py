import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error

class NeuralNetwork:
    pass


class Neuron:
    def __init__(self, numFeatures:int, learningRate:float, nEpochs:int):
        self.weights = pd.Series([random.random * 2 - 1 for feature in range(numFeatures)])
        self.n = learningRate
        self.nEpochs = nEpochs
        pass

    def activation(self, input:float):
        return input

    def predict(self, X:pd.DataFrame):
        return X.apply(lambda x: self.activation(np.sum(self.weights * x)))

    def evaluate(self, X_test:pd.DataFrame, y_test:pd.Series):
        predictions = self.predict(X_test) 
        return mean_squared_error(y_test, predictions), mean_absolute_error(y_test, predictions)


    # new weight[i] = old weight[i] - learning rate (n) * (actual - target) * input[i]

    def train(self, X_train:pd.DataFrame, y_train:pd.Series):
        accuracy = []
        for epoch in range(self.nEpochs):
            predictions = self.predict(X_train)
            
            # w[i] = w[i] - n * (y - t) * x[i]
            self.weights = self.weights - self.n*(predictions - y_train) * X_train

            # calculate error
            acc = (mean_squared_error(y_train, predictions), mean_absolute_error(y_train, predictions))
            accuracy.append(acc)
            
            print(f'epoch#{epoch}: MSE = {acc[0]}, MAE = {acc[1]}')
        return accuracy


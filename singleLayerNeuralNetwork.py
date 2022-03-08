import pandas as pd
import numpy as np
import preprocessing
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error

class NeuralNetwork:
    pass


class Neuron:
    def __init__(self, numFeatures:int, learningRate:float, nEpochs:int):
        self.weights = np.array([random.random() * 2 - 1 for feature in range(numFeatures)])
        self.n = learningRate
        self.nEpochs = nEpochs
        pass

    def activation(self, input:float):
        return input

    def predict(self, X:pd.DataFrame):
        print(self.weights)
        print(X.iloc[0])
        print(self.weights.shape)
        print(X.iloc[0].shape)
        return X.apply(lambda row: self.activation(self.weights.dot(X.iloc[0].values)), axis=1)

    def evaluate(self, X_test:pd.DataFrame, y_test:pd.Series):
        predictions = self.predict(X_test) 
        return mean_squared_error(y_test, predictions), mean_absolute_error(y_test, predictions)


    # new weight[i] = old weight[i] - learning rate (n) * (actual - target) * input[i]

    def train(self, X_train:pd.DataFrame, X_val:pd.DataFrame, y_train:pd.Series, y_val:pd.Series):
        accuracy = []
        for epoch in range(self.nEpochs):
            predictions = self.predict(X_train)
            # print(predictions)
            val_predictions = self.predict(X_val)
            
            # w[i] = w[i] - n * (y - t) * x[i]
            print(self.weights)
            for i, row in X_train.iterrows():
                self.weights = self.weights - self.n*(predictions - y_train.values) * row.values

            # print(predictions.shape)
            # print(val_predictions.shape)
            # print(y_train.shape)
            # print(y_val.shape)

            # calculate error
            acc = (mean_squared_error(y_train, predictions), mean_absolute_error(y_train, predictions), 
                mean_squared_error(y_val, val_predictions), mean_absolute_error(y_val, val_predictions))

            accuracy.append(acc)
            
            print(f'epoch#{epoch}: MSE = {acc[0]}, MAE = {acc[1]}, val_MSE = {acc[2]}, val_MAE = {acc[3]}')
        return accuracy


dt = preprocessing.importData()
X = dt.drop(columns=['ETH'])
y = dt['ETH']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 0)

# print(X_train)
# print(X_val)
# print(y_train)
# print(y_val)

neuron = Neuron(len(X.columns), .3, 50)
neuron.train(X_train, X_val, y_train, y_val)
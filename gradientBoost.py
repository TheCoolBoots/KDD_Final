import pandas as pd
import numpy as np
import preprocessing
import copy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics
from sklearn.ensemble import GradientBoostingRegressor as GBR

class GradientBoostingRegressor:
    def __init__(self, learningRate = .1, n_estimators=100, max_depth=2, loss_function="mean_squared_error"):
        '''
            loss_functions:
            max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, 
            r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_absolute_percentage_error
        '''
        self.learningRate = learningRate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.loss_function = loss_function
        self.loss_functions = {
            "max_error" : sklearn.metrics.max_error,
            "mean_absolute_error" : sklearn.metrics.mean_absolute_error,
            "mean_squared_error" : sklearn.metrics.mean_squared_error,
            "mean_squared_log_error" : sklearn.metrics.mean_squared_log_error,
            "median_absolute_error" : sklearn.metrics.median_absolute_error,
            "r2_score" : sklearn.metrics.r2_score,
            "mean_poisson_deviance" : sklearn.metrics.mean_poisson_deviance,
            "mean_gamma_deviance" : sklearn.metrics.mean_gamma_deviance,
            "mean_absolute_percentage_error" : sklearn.metrics.mean_absolute_percentage_error
        }

    def fit(self, X, y, split_size=0.25):
        X_train, X_test, t_train, t_test = train_test_split(X, y, test_size=split_size)
        X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=split_size)
        trees,train_errors,val_errors = self.make_trees_boost(X_train2, X_val, t_train2, t_val)
        self.trees = self.cut_trees(trees,val_errors)


    def make_trees_boost(self, Xtrain, Xval, ytrain, yval):
        ytrain_orig = copy.deepcopy(ytrain)

        # initial tree
        h = DecisionTreeRegressor(max_depth=self.max_depth)
        h.fit(Xtrain, ytrain)
        ytrain_pred = h.predict(Xtrain)
        yval_pred = h.predict(Xval)
        
        trees = [h]
        train_errors = [self.loss_functions[self.loss_function](ytrain_orig, ytrain_pred)] # the root mean square errors for the validation dataset
        val_errors = [self.loss_functions[self.loss_function](yval, yval_pred)] # the root mean square errors for the validation dataset

        for i in range(self.n_estimators):
            h = DecisionTreeRegressor(max_depth=self.max_depth)
            residual = ytrain - ytrain_pred
            h.fit(Xtrain, residual)
            ytrain_pred = ytrain_pred + self.learningRate * h.predict(Xtrain)
            yval_pred = yval_pred + self.learningRate * h.predict(Xval)
            
            trees.append(h)
            train_errors.append(self.loss_functions[self.loss_function](ytrain_orig, ytrain_pred))
            val_errors.append(self.loss_functions[self.loss_function](yval, yval_pred))
            
        return trees,train_errors,val_errors

    def cut_trees(self, trees,val_errors):
        # Your solution here that finds the minimum validation score and uses only the trees up to that
        min_val = min(val_errors)
        idx_min_val = val_errors.index(min_val)

        return trees[:idx_min_val + 1]

    def predict(self, X):
        tree_predictions = []
        for tree in self.trees:
            tree_predictions.append(tree.predict(X).tolist())
        return np.array(pd.DataFrame(tree_predictions).sum().values.flat)


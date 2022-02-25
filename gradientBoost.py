'''
Modify gradient boosting function we did for lab 5
to accept 
- initial learner to boost
- learning rate
- max trees
- max tree depth 
'''

import pandas as pd
import numpy as np
import preprocessing
import copy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor as GBR

class GradientBoostingRegressor:
    def __init__(self, learningRate = .1, n_estimators=100, max_depth=2):
        self.learningRate = learningRate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y, split_size=0.25):
        X_train, X_test, t_train, t_test = train_test_split(X, y, test_size=split_size)
        X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=split_size)
        trees,train_RMSEs,val_RMSEs = self.make_trees_boost(X_train2, X_val, t_train2, t_val)
        self.trees = self.cut_trees(trees,val_RMSEs)


    def make_trees_boost(self, Xtrain, Xval, ytrain, yval):
        ytrain_orig = copy.deepcopy(ytrain)

        # initial tree
        h = DecisionTreeRegressor(max_depth=self.max_depth)
        h.fit(Xtrain, ytrain)
        ytrain_pred = h.predict(Xtrain)
        yval_pred = h.predict(Xval)
        
        trees = [h]
        train_RMSEs = [np.sqrt(((ytrain_orig - ytrain_pred))**2).sum()/len(ytrain_orig)] # the root mean square errors for the validation dataset
        val_RMSEs = [np.sqrt(((yval - yval_pred))**2).sum()/len(yval)] # the root mean square errors for the validation dataset

        for i in range(self.n_estimators):
            h = DecisionTreeRegressor(max_depth=self.max_depth)
            residual = ytrain - ytrain_pred
            h.fit(Xtrain, residual)
            ytrain_pred = ytrain_pred + self.learningRate * h.predict(Xtrain)
            yval_pred = yval_pred + self.learningRate * h.predict(Xval)
            
            trees.append(h)
            train_RMSEs.append(np.sqrt(((ytrain_orig - ytrain_pred))**2).sum()/len(ytrain_orig))
            val_RMSEs.append(np.sqrt(((yval - yval_pred))**2).sum()/len(yval))
            
        return trees,train_RMSEs,val_RMSEs

    def cut_trees(self, trees,val_RMSEs):
        # Your solution here that finds the minimum validation score and uses only the trees up to that
        min_val = min(val_RMSEs)
        idx_min_val = val_RMSEs.index(min_val)

        return trees[:idx_min_val + 1]

    def predict(self, X):
        tree_predictions = []
        for tree in self.trees:
            tree_predictions.append(tree.predict(X).tolist())
        return np.array(pd.DataFrame(tree_predictions).sum().values.flat)


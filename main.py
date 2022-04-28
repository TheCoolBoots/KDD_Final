import preprocessing
from linearRegModel import LinearRegressionModel
from gradientBoost import GradientBoostingRegressor
from singleLayerNeuralNetwork import Neuron

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

dt = preprocessing.importData()
X = dt.drop(columns=['ETH'])
y = dt['ETH']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)

# linear regression
lr = LinearRegressionModel()
lr.fit(X_train, y_train)
lrmse, lrmae = lr.evaluate(X_test, y_test)
print(lrmse, lrmae)
# (0.005842057982067785, 0.06341566699574394)

# gradient boost
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
mse = mean_squared_error(y_test, gbr.predict(X_test))
mae = mean_absolute_error(y_test, gbr.predict(X_test))
print(mse, mae)
# (0.008870763326737162, 0.07429242957227856)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=13)

neuron = Neuron(len(X.columns), .1, 30)
neuron.train(X_train, X_val, y_train, y_val)
neuron_mse, neuron_mae = neuron.evaluate(X_test, y_test)
print(neuron_mse, neuron_mae)
# (0.0051727149007336735, 0.059702093865324116)

linear reg: 0.008870763326737162 0.07429242957227856
gradient boost: 0.017508403327882914 0.09602076472789384
neuron: 0.0044606277006398725 0.05462005530327893

keras neural net MSE:  0.004314650781452656
keras neural net MAE:  0.05020284280180931
Mean squared error from linear regression:  0.004245044219031201
Mean absolute error from linear regression:  0.05040675085245912
Mean squared error using Random Forest:  0.005457400147313827
Mean absolute error Using Random Forest:  0.056714403083028074
Mean squared error using Gradient Boosting:  0.005602790092401546
Mean absolute error using Gradient Boosting:  0.05687843214611475
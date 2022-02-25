import preprocessing
from linearRegModel import LinearRegressionModel
from gradientBoost import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dt = preprocessing.importData()
X = dt.drop(columns=['ETH']).to_numpy()[:-1]
y = dt['ETH'].to_numpy()[1:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)

# linear regression
lr = LinearRegressionModel(nEpochs=100)
lr.fit(X_train, y_train)

# gradient boost
reg = GradientBoostingRegressor()
reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))

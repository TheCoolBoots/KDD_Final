import preprocessing
from linearRegModel import LinearRegressionModel
from gradientBoost import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor as sklearn_GradientBoostingRegressor
from sklearn.linear_model import LinearRegression as sklearn_LinearRegression
from sklearn.metrics import mean_squared_error

dt = preprocessing.importData()
X = dt.drop(columns=['ETH']).to_numpy()[:-1]
y = dt['ETH'].to_numpy()[1:]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)

# linear regression
lr = LinearRegressionModel()
lr.fit(X_train, y_train)

# gradient boost
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
mse = mean_squared_error(y_test, gbr.predict(X_test))


# sklearn
sklearn_lr = sklearn_LinearRegression()
sklearn_lr.fit(X_train, y_train)

sklearn_gbr = sklearn_GradientBoostingRegressor()
sklearn_gbr.fit(X_train, y_train)
sklearn_mse = mean_squared_error(y_test, sklearn_gbr.predict(X_test))

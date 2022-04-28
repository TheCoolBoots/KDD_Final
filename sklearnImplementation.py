import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot as plt
import preprocessing


dt = preprocessing.importData()
X = dt.drop(columns=['ETH'])
y = dt['ETH']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Scale data so training is faster
scaler=StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# create neural network
kerasNN = Sequential()
kerasNN.add(Dense(128, input_dim=len(X.columns), activation='sigmoid'))
kerasNN.add(Dense(64, activation='sigmoid'))
kerasNN.add(Dense(1, activation='linear'))

kerasNN.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
kerasNN.summary()

history = kerasNN.fit(X_train_scaled, y_train, validation_split=0.2, epochs =20)

#plot the training and validation accuracy and loss at each epoch
kerasLoss = history.history['loss']
kerasValLoss = history.history['val_loss']
epochs = range(len(kerasLoss))
plt.plot(epochs, kerasLoss, 'y', label='Training loss')
plt.plot(epochs, kerasValLoss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

kerasMAE = history.history['mae']
kerasValMAE = history.history['val_mae']
plt.plot(epochs, kerasMAE, 'y', label='Training MAE')
plt.plot(epochs, kerasValMAE, 'r', label='Validation MAE')
plt.title('Training and validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# predictions = model.predict(X_test_scaled)

# Neural network
mse_neural, mae_neural = kerasNN.evaluate(X_test_scaled, y_test)
print('keras neural net MSE: ', mse_neural)
print('keras neural net MAE: ', mae_neural)

# Linear regression
sklearnLR = LinearRegression()
sklearnLR.fit(X_train_scaled, y_train)
slkearnLR_pred = sklearnLR.predict(X_test_scaled)
sklearnLRMSE = mean_squared_error(y_test, slkearnLR_pred)
sklearnLRMAE = mean_absolute_error(y_test, slkearnLR_pred)
print('Mean squared error from linear regression: ', sklearnLRMSE)
print('Mean absolute error from linear regression: ', sklearnLRMAE)

# Random forest
sklearnRandomForest = RandomForestRegressor(n_estimators = 30, random_state=30)
sklearnRandomForest.fit(X_train_scaled, y_train)
sklearnRF_pred = sklearnRandomForest.predict(X_test_scaled)

mse_RF = mean_squared_error(y_test, sklearnRF_pred)
mae_RF = mean_absolute_error(y_test, sklearnRF_pred)
print('Mean squared error using Random Forest: ', mse_RF)
print('Mean absolute error Using Random Forest: ', mae_RF)

# Gradient Boosting
sklearnGB = GradientBoostingRegressor()
sklearnGB.fit(X_train_scaled, y_train)
sklearnGB_pred = sklearnGB.predict(X_test_scaled)
sklearnGBMSE = mean_squared_error(y_test, sklearnGB_pred)
sklearnGBMAE = mean_absolute_error(y_test, sklearnGB_pred)
print('Mean squared error using Gradient Boosting: ', sklearnGBMSE)
print('Mean absolute error using Gradient Boosting: ', sklearnGBMAE)


# Feature Importance
feature_list = list(X.columns)
feature_imp = pd.Series(kerasNN.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:45:03 2018

@author: dimitri.cabaud
"""
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import chdir
chdir("C:\\Users\\dimitri.cabaud\Desktop\Prediction_Icare")
seed = 7 # Fix random seed for reproducibility
np.random.seed(seed)

### Import Data
data_train = pd.read_csv('learning.csv', delimiter=";")
data_test =pd.read_csv('test.csv', delimiter=";")
dataset = pd.concat([data_train,data_test])

y_train = data_train['montant_codif_ht']
y_test = data_test['montant_codif_ht']

dataset_dummies = pd.get_dummies(dataset, columns=['lib_co_mode_gestion_prestation_n', 'energie_n', 'genre_n','marque_majoritaire_n', 'dept_n'])
dataset_dummies = dataset_dummies.drop(['montant_codif_ht', 'dept_n_-1.0', 'lib_co_mode_gestion_prestation_n_0','energie_n_1.0', 'genre_n_1.0','marque_majoritaire_n_1.0', 'dept_n_1.0'], axis=1) 
X_train = dataset_dummies[:317842]
X_test = dataset_dummies[317842:]

description = dataset_dummies.describe()

### Splitting the dataset into the Training set and Test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0, random_state = 0)

### Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


### Tuning the ANN
def build_regressor_for_grid(regressor, nb_units):
    regressor = Sequential()
    regressor.add(Dense(units = nb_units, kernel_initializer = 'uniform', activation = 'relu', input_dim = 325))
    regressor.add(Dense(units = nb_units, kernel_initializer = 'uniform', activation = 'relu'))
    regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    regressor.compile(optimizer = 'adam', loss = 'mae', metrics = ['mse', 'mae', 'mape'])
    return regressor

grid_regressor = KerasRegressor(build_fn = build_regressor_for_grid)
parameters = {'batch_size': [30, 50, 100],
              'epochs': [10, 30],
              'regressor': ['adam'], 
              'nb_units': [100, 150, 200]}
grid_search = GridSearchCV(estimator = grid_regressor,
                           param_grid = parameters)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

### Build one ANN
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units = 166, kernel_initializer = 'uniform', activation = 'relu', input_dim = 331))
    regressor.add(Dense(units = 166, kernel_initializer = 'uniform', activation = 'relu'))
    regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    regressor.compile(optimizer = 'adam', loss = 'mae', metrics = ['mse', 'mae', 'mape'])
    return regressor
regressor_test = KerasRegressor(build_fn = build_regressor, batch_size = 30, epochs = 5)
history = regressor_test.fit(X_train, y_train, validation_split=0.2, batch_size = 30, epochs = 5)

### Evaluate the model
## Plot model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


## Cross validation Score
kfold = KFold(n_splits=10, random_state=seed)
accuracies = cross_val_score(estimator = regressor_test, X = X_train, y = y_train, cv = kfold, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
print("Results: %.2f (%.2f) MSE" % (mean, variance))

## Test on X_test
y_pred = regressor_test.predict(X_test)
mean_absolute_error(y_test, y_pred)




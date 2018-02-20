# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:45:03 2018

@author: dimitri.cabaud
"""
from xgboost.sklearn import XGBRegressor  
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st
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

y_train = data_train['montant_codif_ht']
y_test = data_test['montant_codif_ht']

X_train = data_train
X_train.drop(['montant_codif_ht'], axis=1, inplace=True)
X_test = data_test
X_test.drop(['montant_codif_ht'], axis=1, inplace=True)


### Splitting the dataset into the Training set and Test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0, random_state = 0)

### Feature Scaling
"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
### Tuning 
one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}

xgbreg = XGBRegressor(nthreads=-1)  

gs = RandomizedSearchCV(xgbreg, params, n_jobs=1)  
gs.fit(X_train, y_train)  
gs.best_model_  

### Build the model


### Evaluate the model


## Test on X_test

mean_absolute_error(y_test, y_pred)




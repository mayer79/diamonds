#======================================================================
# Regression Examples
#======================================================================

import os

os.chdir("C:/projects/diamonds")

#======================================================================
# Data prep 
#======================================================================

import numpy as np
import pandas as pd

diamonds = pd.read_csv("data/diamonds.csv")
diamonds.drop([diamonds.columns[0], "x", "y", "z"], axis=1, inplace=True)  
diamonds.head()

cut_map = {"Fair": 1,
           "Good": 2,
           "Very Good": 3,
           "Premium": 4,
           "Ideal": 5}

clarity_map = {"I1": 1,
               "SI2": 2,
               "SI1": 3,
               "VS2": 4,
               "VS1": 5,
               "VVS2": 6,
               "VVS1": 7,
               "IF": 8}


diamonds['cut'] = diamonds['cut'].map(cut_map)
diamonds['clarity'] = diamonds['clarity'].map(clarity_map)
diamonds["color"] = diamonds["color"].factorize(sort=True)[0]
diamonds[["log_price", "log_carat"]] = np.log(diamonds[["price", "carat"]])

# Train/test split
x = ("log_carat", "cut", "color", "clarity", "depth", "table")
y = "log_price"

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diamonds.loc[:, x], 
                                                    diamonds.loc[:, y], 
                                                    test_size = 0.15, 
                                                    random_state = 37364634)


from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor


#======================================================================
# Random forest
#======================================================================

rf = RandomForestRegressor(500, n_jobs=8, oob_score=True)
rf.fit(X_train, y_train)
print("OOB R-squared: %f" % rf.oob_score_) # 0.988
# rf.score(X_test, y_test)  # r-squared 0.9876


#======================================================================
# XGB 
#======================================================================

import xgboost as xgb

model_xgb = xgb.XGBRegressor(learning_rate=0.01, max_depth=8, n_estimators=850,
                             reg_lambda=0.2,
                             objective="reg:linear",
                             subsample=0.7, random_state =7, nthread = 7)

from sklearn.model_selection import cross_val_score

# CV
result = cross_val_score(model_xgb, X_train, y_train, cv=5)
print("Average R-squared over all folds: %.3f" % result.mean())


#======================================================================
# LGB 
#======================================================================

import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',
                              num_leaves=63,
                              learning_rate=0.02, 
                              n_estimators=850,
                              bagging_fraction = 0.7,
                              bagging_freq = 1, 
                              feature_fraction = 1,
                              bagging_seed=9,
                              min_data_in_leaf=20, 
                              max_depth=14, 
                              n_jobs=7)

# CV
result = cross_val_score(model_lgb, X_train, y_train, cv=5)
print('Average R-squared is %.3f' %result.mean()) # Average accuracy is 0.991

# Randomized grid search CV for LGB
from sklearn.model_selection import RandomizedSearchCV

param_grid = dict(bagging_fraction = [0.6, 0.8, 0.9],
                  feature_fraction = [0.6, 0.8, 1],
                  min_data_in_leaf = [10, 50, 100, 200],
                  n_estimators     = [400, 800, 1200],
                  num_leaves       = [7, 15, 31, 63])

myGrid = RandomizedSearchCV(model_lgb, param_distributions=param_grid, cv=5)
results = myGrid.fit(X_train, y_train)
results.best_params_
results.best_score_ # 0.9905

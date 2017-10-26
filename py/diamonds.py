#======================================================================
# Regression Examples for "partialPlot"
#======================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from glmnet_py import glmnet
from ggplot import diamonds # for data set "diamonds"
from sklearn.model_selection import train_test_split
# source("../partialPlot/R/partialPlot.R") # or your path

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#======================================================================
# Data prep 
#======================================================================

diamonds.head()

diamonds[["log_price", "log_carat"]] = np.log(diamonds[["price", "carat"]])
diamonds["cut"] = 1*(diamonds.cut == "Fair") + 2*(diamonds.cut == "Good") + 3*(diamonds.cut == "Very Good") + 4*(diamonds.cut == "Premium") + 5*(diamonds.cut == "Ideal")
diamonds["color"] = diamonds["color"].factorize(sort=True)[0]
diamonds["clarity"] = 1*(diamonds.clarity == "I1") + 2*(diamonds.clarity == "SI2") + 3*(diamonds.clarity == "SI1") + 4*(diamonds.clarity == "VS2") + 5*(diamonds.clarity == "VS1") + 6*(diamonds.clarity == "VVS2") + 7*(diamonds.clarity == "VVS1") + 8*(diamonds.clarity ==  "IF")

# Train/test split
x = ("log_carat", "cut", "color", "clarity", "depth", "table")

X_train, X_test, y_train, y_test = train_test_split(diamonds.loc[:, x], 
                                                    diamonds.log_price, 
                                                    test_size = 0.15, 
                                                    random_state = 37364634)

#======================================================================
# Small functions
#======================================================================

# Calculate R squared
def r2(y, pred):
  return 1 - np.var(y - pred, ddof=1) / np.var(y, ddof=1)  


def rmse(y, pred):
  return np.sqrt(np.mean((y - pred)**2))


# Show all partial dependency plots
partialDiamondsPlot <- function(fit) {
  par(mfrow = 3:2,
      oma = c(0, 0, 0, 0) + 0.3,
      mar = c(4, 2, 0, 0) + 0.1,
      mgp = c(2, 0.5, 0.5))
  
  partialPlot(fit, train$X, xname = "log_carat")
  partialPlot(fit, train$X, xname = "cut", discrete.x = TRUE)
  partialPlot(fit, train$X, xname = "color", discrete.x = TRUE)
  partialPlot(fit, train$X, xname = "clarity", discrete.x = TRUE)
  partialPlot(fit, train$X, xname = "depth")
  partialPlot(fit, train$X, xname = "table")
}


#======================================================================
# OLS
#======================================================================

fit_ols = glmnet.glmnet(x = X_train.values, y = y_train.values)
fit_ols
pred <- predict(fit_ols, test$X)
r2(pred, test$y) # 97.87%
rmse(pred, test$y) # 14.6%
rmse(predict(fit_ols, train$X), train$y) # 14.6


#======================================================================
# random forest
#======================================================================

fit_ranger <- ranger(log_price ~ log_carat + cut + color + clarity + depth + table, 
                     data = trainDF, importance = "impurity", num.trees = 500, 
                     always.split.variables = "log_carat", seed = 837363) 
fit_ranger # Estimated R2 0.9887582 

r2(test$y, predict(fit_ranger, testDF)$predictions) # 0.9889626
plot(importance(fit_ranger))
object.size(fit_ranger) # 300 MB

# Effects plots
partialDiamondsPlot(fit_ranger)


#======================================================================
# gradient boosting with former king "XGBoost"
#======================================================================

dtrain = xgb.DMatrix(X_train.values, label = y_train.values)
dtest = xgb.DMatrix(X_test.values, label = y_test.values)
watchlist = [dtrain, dtest]

param = {"max_depth": 8, 
         "learning_rate": 0.01, 
         "nthread": 7, 
         "lambda": 0.2, 
         "objective": "reg:linear", 
         "eval_metric": "rmse", 
         "subsample": 0.7}

%time fit_xgb = xgb.train(param, dtrain, num_boost_round = 850)
r2(y_test, fit_xgb.predict(dtest)) # 0.9906

# partialDiamondsPlot(fit_xgb)


#======================================================================
# gradient boosting with new king "lightGBM"
#======================================================================

dtrain = lgb.Dataset(X_train, label = y_train)
dtest = lgb.Dataset(X_test, label = y_test)

params = {"objective": "regression", 
           "metric": "l2",
           "learning_rate": 0.01,
           "num_leaves": 127,
           "min_data_in_leaf": 20,
           "nthread": 7}

%time fit_lgb = lgb.train(params, dtrain, num_boost_round = 850)
r2(y_test, fit_lgb.predict(X_test)) # 0.990430205807542


partialDiamondsPlot(fit_lgb)

#======================================================================
# Tuned by gridsearch CV
#======================================================================

paramGrid <- expand.grid(iteration = NA_integer_, # filled by algorithm
                         score = NA_real_,     # "
                         learning_rate = c(0.05, 0.01),
                         num_leaves = 2^(6:8) - 1,
                         lambda_l1 = c(0, 0.2),
                         min_data_in_leaf = 20,
                         max_depth = 14,
                         feature_fraction = 1,
                         bagging_fraction = c(0.7, 1),
                         bagging_freq = 4,
                         nthread = 4)

(n <- nrow(paramGrid)) # 108

for (i in seq_len(n)) {
  print(i)
  gc(verbose = FALSE) # clean memory
  
  cvm <- lgb.cv(as.list(paramGrid[i, -(1:2)]), 
                dtrain,     
                nrounds = 1000, # we use early stopping
                nfold = 5,
                objective = "regression",
                showsd = FALSE,
                early_stopping_rounds = 50,
                verbose = 0L)
  
  paramGrid[i, 1:2] <- as.list(cvm)[c("best_iter", "best_score")]
  save(paramGrid, file = "paramGrid.RData") # if lgb crashes
}

# load("paramGrid.RData", verbose = TRUE)
head(paramGrid <- paramGrid[order(-paramGrid$score), ])

# Use best m
m <- 5

# keep test predictions, no model
predList <- vector(mode = "list", length = m)

for (i in seq_len(m)) {
  print(i)
  gc(verbose = FALSE) # clean memory
  
  fit_temp <- lgb.train(paramGrid[i, -(1:2)], 
                        data = dtrain, 
                        nrounds = paramGrid[i, "iteration"],
                        objective = "regression",
                        verbose = 0L)
  
  predList[[i]] <- predict(fit_temp, test$X)
}

pred <- rowMeans(do.call(cbind, predList))
r2(test$y, pred) # 0.99125


#======================================================================
# Regression Examples for "partialPlot"
#======================================================================

import numpy as np
import pandas as pd
from ggplot import diamonds # for data set "diamonds"
# source("../partialPlot/R/partialPlot.R") # or your path

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


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


#======================================================================
# Tuned by gridsearch CV
#======================================================================

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

model_rf = RandomForestRegressor(n_estimators=500, max_features=2, n_jobs=4, min_samples_split=5, max_depth=20)
%time model_rf.fit(X_train, y_train)
r2(y_test, model_rf.predict(X_test))

model_xgb = xgb.XGBRegressor(learning_rate=0.01, max_depth=8, n_estimators=850,
                             reg_lambda=0.2,
                             objective="reg:linear",
                             subsample=0.7, random_state =7, nthread = 7)


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=63,
                              learning_rate=0.01, n_estimators=850,
                              bagging_fraction = 0.7,
                              bagging_freq = 4, feature_fraction = 1,
                              bagging_seed=9,
                              min_data_in_leaf =20, max_depth=14, n_jobs=7)

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_rf) # 0.1102 (0.0008)
print("\nLGB score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb) # 0.0948 (0.0005)
print("\nXGB score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb) # 0.0959 (0.0008)
print("\nLGB score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
    
averaged_models = AveragingModels(models = (model_xgb, model_lgb))
score = rmsle_cv(averaged_models) # Averaged base models score: 0.0950 (0.0007)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
    
    
stacked_averaged_models = StackingAveragedModels(base_models = (model_xgb, model_lgb),
                                                 meta_model = ENet)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# Create a pipeline that standardizes the data then creates a model
import numpy as np
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import expon as sp_expon

# load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# create pipeline
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=4))

# CV
result = cross_val_score(model, X, Y, cv=10)
print('Average accuracy is %.3f' %result.mean())

# Grid search
param_grid = dict(randomforestclassifier__max_features=[1, 2, 3, 4])
myGrid = GridSearchCV(model, param_grid=param_grid, cv=5)
results = myGrid.fit(X, Y)
print('Best score is %0.3f and achieved by %s' % (results.best_score_, results.best_params_))

# Random search
param_grid = dict(pca__n_components=np.arange(1, 8), lr__C=sp_expon(1))
myGrid = RandomizedSearchCV(model, param_distributions=param_grid, cv=kfold)
results = myGrid.fit(X, Y)
results.best_params_
results.best_score_

#======================================================================
# Regression Examples for "partialPlot"
#======================================================================

library(glmnet)
library(ggplot2) # for data set "diamonds"
library(xgboost)
library(lightgbm)
library(ranger)
source("../partialPlot/R/partialPlot.R") # or your path


#======================================================================
# Data prep 
#======================================================================

diamonds <- transform(as.data.frame(diamonds),
                      log_price = log(price),
                      log_carat = log(carat),
                      cut = as.numeric(cut),
                      color = as.numeric(color),
                      clarity = as.numeric(clarity))

# Train/test split
set.seed(3928272)
.in <- sample(c(FALSE, TRUE), nrow(diamonds), replace = TRUE, p = c(0.15, 0.85))

x <- c("log_carat", "cut", "color", "clarity", "depth", "table")

train <- list(y = diamonds$log_price[.in],
              X = as.matrix(diamonds[.in, x]))
test <- list(y = diamonds$log_price[!.in],
             X = as.matrix(diamonds[!.in, x]))

trainDF <- diamonds[.in, ]
testDF <- diamonds[!.in, ]

#======================================================================
# Small functions
#======================================================================

# Calculate R squared
r2 <- function(y, pred) {
  1 - var(y - pred) / var(y)  
}

rmse <- function(y, pred) {
  sqrt(mean((y - pred)^2))
}

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

fit_ols <- glmnet(x = train$X, y = train$y, lambda = 0)
fit_ols
pred <- predict(fit_ols, test$X)
r2(pred, test$y) # 97.87%
rmse(pred, test$y) # 14.6%
rmse(predict(fit_ols, train$X), train$y) # 14.6


#======================================================================
# One single tree
#======================================================================

fit_tree <- rpart(formula = reformulate(x, response = "log_price"), data = trainDF, cp = 0)
plot(fit_tree)
text(fit_tree)
pred <- predict(fit_tree, testDF)
r2(pred, test$y) # 98.8%
rmse(pred, test$y) # 0.108


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

dtrain <- xgb.DMatrix(train$X, label = train$y)
dtest <- xgb.DMatrix(test$X, label = test$y)
watchlist <- list(train = dtrain, test = dtest)

param <- list(max_depth = 8, 
              learning_rate = 0.01, 
              nthread = 2, 
              lambda = 0.2, 
              objective = "reg:linear", 
              eval_metric = "rmse", 
              subsample = 0.7)

fit_xgb <- xgb.train(param, 
                     dtrain, 
                     watchlist = watchlist, 
                     nrounds = 850, 
                     early_stopping_rounds = 5)
r2(test$y, predict(fit_xgb, test$X)) # 0.99132

partialDiamondsPlot(fit_xgb)


#======================================================================
# gradient boosting with new king "lightGBM"
#======================================================================

dtrain <- lgb.Dataset(train$X, label = train$y)
dtest <- lgb.Dataset(test$X, label = test$y)

params <- list(objective = "regression", 
               metric = "l2",
               learning_rate = 0.01,
               num_leaves = 127,
               min_data_in_leaf = 20)

system.time(fit_lgb <- lgb.train(data = dtrain,
                                 params = params, 
                                 nrounds = 850,
                                 verbose = 0L))
r2(test$y, predict(fit_lgb, test$X)) # 0.9912244

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


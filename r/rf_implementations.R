#======================================================================
# Benchmark of most important RF implementations available in R
#======================================================================

library(randomForest)
library(ranger)
library(randomForestSRC)
library(Rborist)
library(h2o)
library(lightgbm)
library(ggplot2)
library(xgboost)

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
y <- "log_price"

train <- list(y = diamonds[.in, y],
              X = as.matrix(diamonds[.in, x]))
test <- list(y = diamonds[!.in, y],
             X = as.matrix(diamonds[!.in, x]))
trainDF <- diamonds[.in, c(y, x)]
testDF <- diamonds[!.in, c(y, x)]

# For XGBoost
dtrain_xgb <- xgb.DMatrix(train$X, label = train$y)
watchlist <- list(train = dtrain_xgb)

# For lightGBM
dtrain_lgb <- lgb.Dataset(train$X, label = train$y)


#======================================================================
# Small function
#======================================================================

# Some performance measures
perf <- function(y, pred) {
  c(r2 = 1 - var(y - pred) / var(y),
    rmse = sqrt(mean((y - pred)^2)),
    mae = mean(abs(y - pred)))
}

mtry <- 2

# Ranger
system.time(fit_ranger <- ranger(reformulate(x, y), 
                                 data = trainDF, 
                                 num.trees = 500, 
                                 min.node.size = 5,
                                 mtry = mtry,
                                 seed = 837363)) # 5.5
pred <- predict(fit_ranger, testDF)$predictions
perf(test$y, pred) # 0.98896
object.size(fit_ranger) # 318 MB
rm(fit_ranger)

# randomForestSRC
system.time(fit_rfsrc <- rfsrc(reformulate(x, y), 
                               data = trainDF, 
                               mtry = mtry,
                               ntree = 500, 
                               nodedepth = 20,
                               nodesize = 5,
                               seed = 837363)) # 15
pred <- predict(fit_rfsrc, testDF)$predicted
perf(test$y, pred) # 0.989343
object.size(fit_rfsrc) # 541 MB
rm(fit_rfsrc)

# randomForest
system.time(fit_rf <- randomForest(reformulate(x, y), 
                                   data = trainDF, 
                                   ntree = 500,
                                   mtry = mtry,
                                   nodesize = 5, 
                                   seed = 837363)) # 712.65 
perf(test$y, predict(fit_rf, testDF)) # 0.9889682
object.size(fit_rf) # 357 MB

# RBORIST
set.seed(345)
system.time(fit_rbor <- Rborist(x = train$X, y = train$y, 
                                nTree = 500, 
                                autoCompress = mtry / length(x),  
                                minInfo = 0,
                                nLevel = 20,
                                minNode = 5)) # 38.64
pred <- predict(fit_rbor, test$X)$yPred
perf(test$y, pred) # 0.988672
object.size(fit_rbor) # 407 MB

# h2o
h2o.init(nthreads = 8)
h2o.train <- as.h2o(trainDF)
h2o.test <- as.h2o(testDF)

system.time(fit_h2o <- h2o.randomForest(x = x, 
                                        y = y, 
                                        training_frame = h2o.train, 
                                        ntrees = 500, 
                                        max_depth = 20,
                                        mtries = mtry,
                                        min_rows = 5,
                                        nbins = 20,
                                        seed = 22342,
                                        min_split_improvement = 0)) # 22
pred <- as.data.frame(predict(fit_h2o, h2o.test))$predict
perf(test$y, pred) # 0.9890738

# xgboost
param <- list(max_depth = 20,
              learning_rate = 1,
              nthread = 4,
              objective = "reg:linear",
              eval_metric = "rmse",
              subsample = 0.63,
              lambda = 0,
              colsample_bylevel = 1/3)

system.time(fit_xgb <- xgb.train(param,
                                 dtrain_xgb,
                                 watchlist = watchlist,
                                 nrounds = 1,
                                 num_parallel_tree = 500,
                                 verbose = 0)) # 21 sec
pred <- predict(fit_xgb, test$X)
perf(test$y, pred) # 0.9892
object.size(fit_xgb) # 459 MB

# lgb
param <- list(learning_rate = 1,
              nthread = 8,
              objective = "regression_l2",
              metric = "rmse",
              bagging_fraction = 0.63,
              bagging_freq = 1,
              feature_fraction = 1/3,
              max_depth = 20,
              num_leaves = 2^10,
              boosting_type = "rf")

system.time(fit_lgb <- lgb.train(param,
                                 dtrain_lgb,
                                 nrounds = 500,
                                 verbose = 0)) # 26.6 sec
pred <- predict(fit_lgb, test$X)
perf(test$y, pred) # 0.9888
object.size(fit_lgb) # 459 MB

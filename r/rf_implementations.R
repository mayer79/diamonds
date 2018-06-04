#======================================================================
# Benchmark of most important RF implementations available in R
#======================================================================

library(tidyverse)
library(randomForest)
library(parallelRandomForest)
library(ranger)
library(randomForestSRC)
library(Rborist)
library(h2o)
library(xgboost)

#======================================================================
# Data prep 
#======================================================================

diamonds <- diamonds %>% 
  mutate_if(is.ordered, as.numeric) %>% 
  mutate(log_price = log(price),
         log_carat = log(carat))

# Train/test split
set.seed(3928272)
.in <- sample(c(FALSE, TRUE), nrow(diamonds), replace = TRUE, p = c(0.15, 0.85))

x <- c("log_carat", "cut", "color", "clarity", "depth", "table")
y <- "log_price"

train <- list(y = diamonds[[y]][.in], 
              X = as.matrix(diamonds[.in, x]))
test <- list(y = diamonds[[y]][!.in],
             X = as.matrix(diamonds[!.in, x]))
trainDF <- diamonds[.in, c(y, x)]
testDF <- diamonds[!.in, c(y, x)]

# For XGBoost
dtrain_xgb <- xgb.DMatrix(train$X, label = train$y)
watchlist <- list(train = dtrain_xgb)

#======================================================================
# Small function
#======================================================================

# Some performance measures
perf <- function(y, pred) {
  res <- y - pred
  c(r2 = 1 - var(res) / var(y),
    rmse = sqrt(mean(res^2)),
    mae = mean(abs(res)))
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
perf(test$y, pred) # 0.989
object.size(fit_ranger) # 318 MB
rm(fit_ranger)

# randomForestSRC
system.time(fit_rfsrc <- rfsrc(reformulate(x, y), 
                               data = trainDF, 
                               mtry = mtry,
                               ntree = 500, 
                               nodedepth = 20,
                               nodesize = 5,
                               seed = 837363)) # 20
pred <- predict(fit_rfsrc, testDF)$predicted
perf(test$y, pred) # 0.989
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
                                minNode = 5)) # 18
pred <- predict(fit_rbor, test$X)$yPred
perf(test$y, pred) # 0.989
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
                                        min_split_improvement = 0)) # 26
pred <- as.data.frame(predict(fit_h2o, h2o.test))$predict
perf(test$y, pred) # 0.989

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
                                 verbose = 0)) # 27 sec
pred <- predict(fit_xgb, test$X)
perf(test$y, pred) # 0.989
object.size(fit_xgb) # 466 MB

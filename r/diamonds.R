#======================================================================
# Regression Examples
#======================================================================

library(tidyverse)
library(glmnet)
library(rpart)
library(ranger)
library(xgboost)
library(lightgbm)
# installation of lightgbm (might cause some headaches, but without GPU okay)
# library(devtools)
# install_github("Microsoft/LightGBM", subdir = "R-package")

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

#======================================================================
# Elastic net (GLM with ridge and/or LASSO penalty)
#======================================================================

# Use cross-validation to find best alpha and lambda, the two penalization parameters of elastic net
for (i in 0:10) {
  fit_ols <- cv.glmnet(x = train$X, 
                       y = train$y, 
                       alpha = i / 10, 
                       nfolds = 5, 
                       type.measure = "mse")
  if (i == 0) cat("\n alpha\t rmse (CV)")
  cat("\n", i / 10, "\t", sqrt(min(fit_ols$cvm)))
}

# Use CV to find best lambda given optimal alpha of 0.2
set.seed(342)
fit_ols <- cv.glmnet(x = train$X, 
                     y = train$y, 
                     alpha = 0.2, 
                     nfolds = 5, 
                     type.measure = "mse")
cat("Best rmse (CV):", sqrt(min(fit_ols$cvm))) # 0.146478

# # On test sample (always with best lambda)
# pred <- predict(fit_ols, test$X)
# perf(test$y, pred)

#======================================================================
# One single tree (just for illustration)
#======================================================================

fit_tree <- rpart(reformulate(x, y), data = trainDF)
plot(fit_tree)
text(fit_tree)
# pred <- predict(fit_tree, testDF)
# perf(test$y, pred) # 0.2929

#======================================================================
# random forest (without tuning - that is their strength ;))
#======================================================================

# Use OOB estimates to find optimal mtry (small number of trees)
for (m in seq_along(x)) {
  fit_rf <- ranger(reformulate(x, y), data = trainDF, num.trees = 50, mtry = m, seed = 37 + m * 3)
  if (m == 1) cat("m\t rmse (OOB)")
  cat("\n", m, "\t", sqrt(fit_rf$prediction.error))
}

# Use optimal mtry to fit 500 trees
m <- 3
fit_rf <- ranger(reformulate(x, y), 
                 data = trainDF, importance = "impurity", num.trees = 500, 
                 mtry = m, seed = 837363)
cat("Best rmse (OOB):", sqrt(fit_rf$prediction.error)) # 0.1035

# Interpretation
imp_rf <- sort(importance(fit_rf))
imp_rf_df <- data.frame(Feature = names(imp_rf), Gain = imp_rf, 
                        row.names = NULL)[rev(seq_along(imp_rf)), ]
par(mar = c(3, 12, 0, 1))
barplot(imp_rf)
object.size(fit_rf) # 424 MB

# perf(test$y, predict(fit_rf, testDF)$predictions) # 0.1018

#======================================================================
# gradient boosting with "XGBoost"
#======================================================================

dtrain_xgb <- xgb.DMatrix(train$X, label = train$y)
watchlist <- list(train = dtrain_xgb)

# Grid search CV (vary different parameters together first to narrow reasonable range)
paramGrid <- expand.grid(iteration = NA_integer_, # filled by algorithm
                         score = NA_real_,     # "
                         learning_rate = c(0.02, 0.05), # c(0.2, 0.1, 0.05, 0.02, 0.01),
                         max_depth = 5:7, # 1:10, -> 5:6
                         min_child_weight = c(0, 1e-04, 1e-2), # c(0, 10^-(-1:4)) -> 1e-04
                         colsample_bytree = c(0.5, 0.7, 0.9), # seq(0.5, 1, by = 0.1), # 
                         subsample = c(0.8, 1), # seq(0.5, 1, by = 0.1), # ok
                         lambda = 0:2, # c(0, 0.1, 0.5, 1, 2, 3), # l2 penalty
                         alpha = 0:2, # c(0, 0.1, 0.5, 1, 2, 3), # l1 penalty
                         min_split_loss = c(0, 1e-04, 1e-02), # c(0, 10^-(-1:4)), 
                         nthread = 3, # ok?
                         eval_metric = "rmse")

(n <- nrow(paramGrid)) # 2916
set.seed(342267)
paramGrid <- paramGrid[sample(n, 10), ]
(n <- nrow(paramGrid)) # 100

for (i in seq_len(n)) { # i = 1
  print(i)
  cvm <- xgb.cv(as.list(paramGrid[i, -(1:2)]), 
                dtrain_xgb,     
                nrounds = 5000, # we use early stopping
                nfold = 5,
                objective = "reg:linear",
                showsd = FALSE,
                early_stopping_rounds = 20,
                verbose = 0L)
  paramGrid[i, 1] <- bi <- cvm$best_iteration
  paramGrid[i, 2] <- as.numeric(cvm$evaluation_log[bi, "test_rmse_mean"])
  save(paramGrid, file = "paramGrid_xgb.RData")
}

# load("paramGrid_xgb.RData", verbose = TRUE)
head(paramGrid <- paramGrid[order(paramGrid$score), ])

# Best only (no ensembling)
cat("Best rmse (CV):", paramGrid[1, "score"]) # 0.096946
fit_xgb <- xgb.train(paramGrid[1, -(1:2)], 
                     data = dtrain_xgb, 
                     nrounds = paramGrid[1, "iteration"] * 1.05,
                     objective = "reg:linear")
pred <- predict(fit_xgb, test$X)
perf(test$y, pred) # 0.99108733

#======================================================================
# gradient boosting with "lightGBM"
#======================================================================

dtrain_lgb <- lgb.Dataset(train$X, label = train$y)

# A grid of possible parameters
paramGrid <- expand.grid(iteration = NA_integer_, # filled by algorithm
                         score = NA_real_,     # "
                         learning_rate = c(0.05, 0.02), # c(1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01), # -> 0.02
                         num_leaves = c(31, 63, 127), # c(31, 63, 127, 255),
                         # max_depth = 14,
                         min_data_in_leaf = c(10, 20, 50),
                         lambda_l1 = c(0, 0.5, 1),
                         lambda_l2 = 2:6, #  c(0, 0.1, 0.5, 1:6),
                         min_sum_hessian_in_leaf = c(0.001, 0.1), # c(0, 1e-3, 0.1),
                         feature_fraction = c(0.7, 1), # seq(0.5, 1, by = 0.1),
                         bagging_fraction = c(0.8, 1), # seq(0.5, 1, by = 0.1),
                         bagging_freq = 4,
                         nthread = 4,
                         max_bin = 255)

(n <- nrow(paramGrid)) # 2160
set.seed(34234)
paramGrid <- paramGrid[sample(n, 100), ]
(n <- nrow(paramGrid)) # 100

for (i in seq_len(n)) {
  print(i)
  cvm <- lgb.cv(as.list(paramGrid[i, -(1:2)]), 
                dtrain_lgb,     
                nrounds = 5000, # we use early stopping
                nfold = 5,
                objective = "regression",
                showsd = FALSE,
                early_stopping_rounds = 20,
                verbose = -1,
                metric = "rmse")
  paramGrid[i, 1:2] <- as.list(cvm)[c("best_iter", "best_score")]
  save(paramGrid, file = "paramGrid_lgb.RData") # if lgb crashes
}

# load("paramGrid_lgb.RData", verbose = TRUE)
head(paramGrid <- paramGrid[order(-paramGrid$score), ])

# Use best only (no ensembling)
cat("Best rmse (CV):", -paramGrid[1, "score"]) # 0.09608951

system.time(fit_lgb <- lgb.train(paramGrid[1, -(1:2)], 
                     data = dtrain_lgb, 
                     nrounds = paramGrid[1, "iteration"] * 1.05,
                     objective = "regression"))

# Interpretation
imp_lgb <- lgb.importance(fit_lgb)
print(imp_lgb)
lgb.plot.importance(imp_lgb, top_n = length(x))

# # Select best and test 
# pred <- predict(fit_lgb, test$X)
# perf(test$y, pred) # 0.09463408

# Now use an average of top 3 models
m <- 3

# keep test predictions, no model
predList <- vector(mode = "list", length = m)

for (i in seq_len(m)) {
  print(i)
  fit_temp <- lgb.train(paramGrid[i, -(1:2)], 
                        data = dtrain_lgb, 
                        nrounds = paramGrid[i, "iteration"] * 1.05,
                        objective = "regression",
                        verbose = -1)
  predList[[i]] <- predict(fit_temp, test$X)
}
pred <- rowMeans(do.call(cbind, predList))
# # Test
# perf(test$y, pred) # 0.99126 R2


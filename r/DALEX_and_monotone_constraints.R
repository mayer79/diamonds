library(xgboost)
library(lightgbm)
library(gbm)
library(ggplot2) # For the data set
library(DALEX) # For model interpretation
library(tidyverse) # For data prep

head(diamonds) 

diamonds <- diamonds %>% 
  mutate_if(is.ordered, as.numeric) %>% 
  mutate(log_price = log(price),
         log_carat = log(carat)) 

x <- c("log_carat", "cut", "color", "clarity", "depth", "table")
y <- "log_price"

param_xgb <- list(max_depth = 5, 
                  eta = 0.05, 
                  nthread = 4, 
                  objective = "reg:linear",
                  monotone_constraints = c(1, 1, 0, 0, 0, 0))

param_lgb <- list(num_leaves = 31, 
                  learning_rate = 0.05, 
                  nthread = 4, 
                  objective = "regression_l2",
                  mc = "1,1,0,0,0,0")

model_matrix_train <- data.matrix(diamonds[, x])
data_train_xgb <- xgb.DMatrix(model_matrix_train, label = diamonds[[y]])
data_train_lgb <- lgb.Dataset(model_matrix_train, label = diamonds[[y]])

xgb_model <- xgb.train(param_xgb, data_train_xgb, nrounds = 200)
lgb_model <- lgb.train(param_lgb, data_train_lgb, nrounds = 200)
gbm_model <- gbm(reformulate(x, y), data = diamonds, n.trees = 200, 
                 interaction.depth = 2, shrinkage = 0.05, 
                 var.monotone = c(1, 1, 0, 0, 0, 0))

# Initializing the "explainer" on a subset of 10'000 obs
set.seed(574)
rand_ind <- sample(1:nrow(model_matrix_train), 10000)
explainer_xgb <- explain(xgb_model, data = model_matrix_train[rand_ind, ], y = diamonds[[y]], label = "xgboost")
explainer_lgb <- explain(lgb_model, data = model_matrix_train[rand_ind, ], y = diamonds[[y]], label = "lgb")
explainer_gbm <- explain(gbm_model, data = diamonds[rand_ind, ], y = diamonds[[y]], label = "gbm",
                         predict_function = function(model, x) predict(model, x, n.trees = 200))
explainers <- list(explainer_xgb, explainer_lgb, explainer_gbm)

# Use explainer to visualize the monotonic relationship between log_carat and log_price
sv_carat <- lapply(explainers, single_variable, variable = "log_carat", type = "pdp")
do.call(plot, sv_carat)

# Visualize the monotonic relationship between cut and log_price
sv_cut <- lapply(explainers, single_variable, variable = "cut", type = "pdp")
do.call(plot, sv_cut)

# Use explainer to visualize the contribution of each variable in predicting the first obs
sp_xgb  <- single_prediction(explainer_xgb, observation = model_matrix_train[1, , drop = FALSE])
sp_lgb  <- single_prediction(explainer_lgb, observation = model_matrix_train[1, , drop = FALSE])
sp_gbm  <- single_prediction(explainer_gbm, observation = diamonds[1, ])

plot(sp_xgb, sp_lgb, sp_gbm) # almost the full effect comes from carat

# Variable importance
vd <- lapply(explainers, variable_importance, type = "raw")
do.call(plot, vd)


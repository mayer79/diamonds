# ==============================================================
# In this script, we show how XGBoost can be used as accelerated
# failure time model. We compare with a parametric model.
# ==============================================================

library(xgboost)
library(survival)

# Create data
set.seed(234)
n <- 10000
X <- cbind(x1 = rexp(n), x2 = runif(n))
y <- exp(X[, "x1"] + 0.2 * X[, "x2"] + rnorm(n))

# Introduce 10% censoring
y_up <- y
y_up[1:(n %/% 10)] <- Inf
data <- data.frame(y = y, y_up = y_up, X)

# Split into 90% for train/valid and 10% for test
ix_test <- sample(n, n %/% 10)

# Train/valid
dtrainval <- xgb.DMatrix(
  data = X[-ix_test, ], 
  label_lower_bound = y[-ix_test], 
  label_upper_bound = y_up[-ix_test]
)

# Split into train and valid
n_left <- nrow(dtrainval)
ix_valid <- sample(n_left, n_left %/% 4)
ix_train <- setdiff(1:n_left, ix_valid)

dtrain <- slice(dtrainval, ix_train)
dvalid <- slice(dtrainval, ix_valid)

# Fit and tune XGBoost model
params <- list(
  learning_rate = 0.01,
  objective = "survival:aft",
  aft_loss_distribution = "normal", # log(y) ~ f(x) + sigma normal
  aft_loss_distribution_scale = 1.1,
  max_depth = 3
)

# Other distributions 
# - "extreme" would correspond to "weibull" AFT
# - "logistic" would correspond to "loglogistic" AFT

fit_temp <- xgb.train(
  params,
  dtrain,  
  nrounds = 5000,
  watchlist = list(valid = dvalid),
  early_stopping_rounds = 20,
  print_every_n = 10,
  verbose = 2
)

# Refit on train/valid
fit <- xgb.train(
  params,
  dtrainval,  
  nrounds = fit_temp$best_iteration,
  verbose = 0
)

# Evaluate
y_test <- y[ix_test]
surv_test <- Surv(y_test, event = is.finite(y_up[ix_test]))
pred_xgb <- predict(fit, X[ix_test, ])
summary(pred_xgb)
summary(y_test)
concordance(surv_test ~ pred_xgb) # 0.7172

# Parametric
fit_parametric <- survreg(
  Surv(y, event = is.finite(y_up)) ~ x1 + x2, 
  data = data,
  subset = -ix_test,
  dist = "lognormal" # log(y) ~ f(x) + sigma normal
)

summary(fit_parametric)
pred_parametric <- predict(fit_parametric, newdata = data[ix_test, ])
summary(pred_parametric)
summary(y_test)
concordance(surv_test ~ pred_parametric) # 0.7178

# SHAP-Analysis (log-prediction scale)
library(SHAPforxgboost)

shap_values <- shap.prep(fit, X_train = X[ix_test, ])
shap.plot.summary(shap_values)
shap.plot.dependence(shap_values, "x1", smooth = FALSE, 
                     color_feature = "auto")
shap.plot.dependence(shap_values, "x2", smooth = FALSE, 
                     color_feature = "auto")

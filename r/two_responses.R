#======================================================================
# Illustration how two responses can be modelled by one single model
#======================================================================

library(tidyverse)
library(ranger)
library(lightgbm)
library(DALEX) # For model interpretation

# devtools::install_url('https://github.com/catboost/catboost/releases/download/v0.10.0/catboost-R-Windows-0.10.0.tgz', args = c("--no-multiarch"))

#======================================================================
# Data prep 
#======================================================================

stratum <- "price_or_cut"
y <- "response"
x <- c("carat", "color", "clarity", "table", "depth", stratum)

diamonds_long <- diamonds %>% 
  mutate_if(is.ordered, factor, ordered = FALSE) %>% 
  mutate(log_price = log(price),
         cut = as.integer(cut)) %>% 
  select(-x, -y, -z, -price) %>% 
  gather(key = "price_or_cut", value = "response", log_price, cut, factor_key = TRUE)
head(diamonds_long)
str(diamonds_long)

# Train/test split
set.seed(3928272)
ind <- caret::createDataPartition(diamonds_long[[y]], p = 0.85, list = FALSE) %>% c

lgb_mapping <- function(data) {
  data %>% 
    select_at(x) %>% 
    data.matrix
}

lgb_predict <- function(model, newdata) {
  predict(model, lgb_mapping(newdata))
}

head(lgb_mapping(diamonds_long))

trainDF <- diamonds_long[ind, ]
testDF <- diamonds_long[-ind, ]

#======================================================================
# Small functions
#======================================================================

rmse <- function(y, pred) {
  sqrt(mean((y - pred)^2))
}

mae <- function(y, pred) {
  mean(abs(y - pred))
}

#======================================================================
# random forest (without tuning - that is their strength ;))
#======================================================================

fit_rf <- ranger(reformulate(x, y), data = trainDF, 
                 importance = "impurity", num.trees = 500, seed = 837363)
cat("Best rmse (OOB):", sqrt(fit_rf$prediction.error))

object.size(fit_rf) # 424 MB

# Log-impurity gains
barplot(fit_rf %>% importance %>% sort)

# rmse(test$y, predict(fit_rf, testDF)$predictions) 

#======================================================================
# gradient boosting with "lightGBM"
#======================================================================

dtrain_lgb <- lgb.Dataset(lgb_mapping(trainDF), label = trainDF[[y]])

load("paramGrid_lgb.RData", verbose = TRUE)
head(paramGrid <- paramGrid[order(-paramGrid$score), ])

# Use best only (no ensembling)
cat("Best rmse (CV):", -paramGrid[1, "score"])

system.time(fit_lgb <- lgb.train(paramGrid[1, -(1:2)], 
                                 data = dtrain_lgb, 
                                 nrounds = paramGrid[1, "iteration"],
                                 objective = "regression"))

# Interpretation
imp_lgb <- lgb.importance(fit_lgb)
print(imp_lgb)
lgb.plot.importance(imp_lgb, top_n = length(x))


#======================================================================
# Performance
#======================================================================

null_model <- trainDF %>% 
  group_by_at(stratum) %>% 
  summarize(`Empty Model` = mean(response))

perf_table <- testDF %>%  
  select_at(c(stratum, "response")) %>% 
  mutate(`Random Forest` = predict(fit_rf, testDF)$pred,
         `Gradient Boosting` = predict(fit_lgb, lgb_mapping(testDF))) %>%
  left_join(null_model) %>% 
  gather(key = "Model", value = "pred", -one_of(stratum, "response")) %>% 
  group_by_at(c("Model", stratum)) %>% 
  summarize(n = n(), RMSE = rmse(response, pred))

ggplot(perf_table, aes_string(y = "RMSE", x = "Model")) +
  geom_bar(aes(fill = Model), stat = "identity", position = "dodge") +
  facet_wrap(reformulate(stratum))

#======================================================================
# DALEX
#======================================================================

y_splitted <- split(testDF[[y]], testDF[[stratum]])
DF_splitted <- split(testDF %>% select(-one_of(y)), testDF[[stratum]])

explainers <- Map(explain, list(fit_lgb, fit_lgb), DF_splitted, y_splitted, 
                  label = as.list(names(y_splitted)), predict_function = list(lgb_predict, lgb_predict))

# Variable importance
vd <- lapply(explainers, variable_importance, type = "difference") %>% 
  lapply(function(z) subset(z, variable != stratum))
do.call(plot, vd)

# PDP
sv_carat <- lapply(explainers, single_variable, variable = "carat")
do.call(plot, sv_carat)

sv_color <- lapply(explainers, single_variable, variable = "color")
do.call(plot, sv_color)

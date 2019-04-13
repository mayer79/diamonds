library(mlr)

# Train/test split
set.seed(3928272)
ind <- caret::createDataPartition(diamonds[[y]], p = 0.85, list = FALSE) %>% c

trainDF <- diamonds[ind, c(x, y)]
testDF <- diamonds[-ind, c(x, y)]

# Define task
price_task <- makeRegrTask("price", trainDF, target = y)
lrn_rf <- makeLearner(cl = "regr.ranger", "ranger")
resample <- makeResampleDesc("CV", iters = 5)
param_set <- makeParamSet(
  makeIntegerParam("mtry", lower = 1, upper = 5),
  makeIntegerParam("num.trees", lower = 1, upper = 2, trafo = function(x) round(10^x)),
  makeDiscreteParam("splitrule", values = c("variance", "extratrees")))
ctrl <- makeTuneControlRandom(maxit = 10)

fit <- train(lrn_rf, price_task)

res <- tuneParams(lrn_rf, 
                  price_task, 
                  resampling = resample, 
                  par.set = param_set,
                  control = ctrl)

lrn_rf_opt <- setHyperPars(lrn_rf, par.vals = res$x)
fit <- train(lrn_rf_opt, price_task)
pred <- predict(fit, newdata = testDF)
performance(pred) # 0.01149185 

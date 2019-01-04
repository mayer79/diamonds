library(tidyverse)
library(ranger)

diamonds <- diamonds %>% 
  mutate(log_price = log(price),
         log_carat = log(carat))

fit <- ranger(log_price ~ log_carat, data = diamonds)
saveRDS(fit, file = "fit.rds")

library(plumber)
r <- plumb("test.R")
r$run(port=8000)

# curl -X GET "http://127.0.0.1:8000/predict?carat=0.4" -H  "accept: application/json"
# curl -H "Content-Type: application/json" -X GET -d "{\"carat\":0.7}" "http://127.0.0.1:8000/predict"
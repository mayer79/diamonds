library(ranger)
library(jsonlite)

fit <- read_rds("fit.rds")

#* @apiTitle Diamond price prediction
#* @apiDescription This API takes as input the weight of a 
#*   diamond in carat and returns its predicted price.

#* @param carat:numeric Weight of diamond in carat
#* @get /predict
#* @serializer unboxedJSON
#* @response 200 Returns the predicted price
predict_price <- function(carat) {
  carat <- as.numeric(carat)
  dat <- data.frame(log_carat = log(carat))
  prediction <- round(exp(predict(fit, dat)$predictions), -1)
  list(price = prediction)
}


#* Log some information about the incoming request
#* @filter logger
function(req){
  cat(as.character(Sys.time()), "-", 
      req$QUERY_STRING,
      req$REQUEST_METHOD, req$PATH_INFO, "-", 
      req$HTTP_USER_AGENT, "@", req$REMOTE_ADDR, "\n")
  plumber::forward()
}

#disable gpu devices
#Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)

Sys.setenv(TZ='America/New_York')


library(keras)
library(tensorflow)
#set seed to get reproducible results, seed wyłącza gpu
#set_random_seed(23)
library(tidyverse)
library(dygraphs)
library(lubridate)
library(here)
library(xts)
library(scales)
library(feather)
#library(kerastuneR)

source(here("SCRIPTS/functions.R"))




#loading 1day snp data
dt_1d <-
  read.csv(here("DATA/NKX.csv"))

snp <-
  dt_1d %>%
  select(5) %>%
  mutate(Close = Close/xts::lag.xts(Close) - 1)


#logarytmiczne
snp <- diff(log(dt_1d$Close))

#transform to matrix and drop NA
data <- data.matrix(snp)
data <- na.omit(data)

summary(data)




#define network hyperparams
filters <- 256
kernel_size <- 3
pool_size = 3
hidden_dims <- 250
dropout_rate <- 0.0
batch_size <- 756
l2_reg <- 0.000001
momentum <-0.1

#define sequence length
n_steps <-10 #dla Dense tu ma być 1

#define number of features (input variables)
n_features <- 1


#custom loss function defintion
made <-function(y_true, y_pred) {
  
  loss <- (-1. * tf$sign(y_true * y_pred) * (tf$abs(y_true)))
  
  # loss = tf$where(y_pred > 0, 
  #                 (-1 * tf$sign(y_true * y_pred) * (tf$abs(y_true))),
  #                 0)
  
  tf$reduce_mean(loss)
}
keras.losses.made = made

#define weigths initializer
initializer = initializer_glorot_uniform(seed = 0)

#set keras and tf seed, w pythonie:
#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)


# a <- c(1,1,1,2) #actual returns
# b <- c(1,1,-1,1) #prediced
# 
# made(a,b)


#reshape data to match expected layer input shape
input_shape = list(n_steps, n_features)


#data <- data.matrix(1:3002)

#define lengths of train and test sets
n_train = 2016 #lub 2000
n_test = 756
n_records = (length(data))

#create empty signals vector
signal <- vector()
eval_list <- vector()
preds_list <- vector()


#library(tensorflow)
#set seed to get reproducible results
#set_random_seed(23) 
#main rolling loop
for (i in seq(n_train, n_records, n_test)){
  
  #define optimizer - it needs to be defined inside loop to avoid model checkpoint filesize expanding
  #opt = optimizer_sgd(lr = 0.3, momentum=0.5, nesterov=TRUE)
  opt = optimizer_adam(lr = 0.5, amsgrad = TRUE) #dla dense wyzsza
  
  
  #crate train set
  train <- matrix(data[((i-n_train)+1):i])
  
  print(length(train))
  
  #calculate max and min values and divide by greater of them (normalize)
  #m <- if (abs(max(train))>abs(min(train))) max(train) else min(train)
  #train <- train/m
  
  #create test set and divide by train min/max value
  test <- (matrix(data[((i+1)-n_steps):((i+n_test))])) 
  test <- na.omit(test)
  #test <- test/m
  
  
  splt <- split_sequence(train,n_steps)
  X <- splt[[1]]
  y <- splt[[2]]
  
  #reshape from [samples, timesteps] to [samples, timesteps, features]
  X <- array(X, dim = c(dim(X)[1], dim(X)[2],n_features))
  
  
  
  #early stopping to stop training when it's not improving and model checkpoint to save the best model
  es <- callback_early_stopping(monitor='val_loss', mode='min', patience=6, verbose=1, restore_best_weights = TRUE)
  mc <- callback_model_checkpoint(here(paste("best_model.h5", sep="")), monitor='val_loss', mode='min', save_best_only=TRUE, verbose=1)
  
  
  #initialize model
  model <- keras_model_sequential()
  
  #create the model
  model %>%
    
    
    
  
  layer_lstm(
    units = 512,
    recurrent_activation = "sigmoid",
    return_sequences = TRUE,
    kernel_regularizer = regularizer_l2(l = l2_reg),
    #activity_regularizer=regularizer_l2(0.0001),
    #recurrent_dropout = (0.02),
    kernel_initializer=initializer,
  ) %>%
    layer_dropout(rate = dropout_rate) %>%

    layer_lstm(
      units = 256,
      recurrent_activation = "sigmoid",
      return_sequences = TRUE,
      kernel_regularizer = regularizer_l2(l = l2_reg),
      #activity_regularizer=regularizer_l2(0.0001),
      #recurrent_dropout = (0.02),
      kernel_initializer=initializer,
    ) %>%
    layer_dropout(rate = dropout_rate) %>%



    layer_lstm(
      units = 128,
      recurrent_activation = "sigmoid",
      kernel_regularizer = regularizer_l2(l = l2_reg),
      kernel_initializer=initializer,

      #recurrent_dropout = (0.02),
      #activity_regularizer=regularizer_l2(0.0001),
      input_shape = input_shape
    ) %>%
    layer_dropout(rate = dropout_rate) %>%
    
    #layer_dense(1, activation = "relu")
    layer_dense(1, activation = "linear")
  
  #compile the model
  model %>% compile(
    loss = made,
    optimizer = opt
    #metrics = "mse"
  )
  
  
  print("trainin the model...")
  
  
  hist_RNN <- model%>%
    fit(
      X,
      y,
      batch_size = batch_size,
      validation_split = 0.375,
      epochs = 190,
      verbose = 2,
      shuffle = FALSE, 
      callbacks = list(mc)
    )
  
  
  
  #load savel model from mc
  #since we added restore_best_weights = TRUE model = saved_model
  saved_model = load_model_hdf5(here('best_model.h5'), compile=FALSE)
  saved_model %>% compile(
    optimizer=opt,
    loss = made
  )
  
  
  
  #train and validate the model on train set 
  
  
  
  #split test data into samples and targets
  splt_t <- split_sequence(test,n_steps)
  Xtest <- splt_t[[1]]
  ytest <- splt_t[[2]]
  
  #reshape test data
  Xtest <- array(Xtest, dim = c(dim(Xtest)[1], dim(Xtest)[2],n_features))
  
  
  print("making predictions...")
  #make predictions on test set
  
  preds <- saved_model %>% predict(
    Xtest,
    verbose = 2
  )
  
  ##add preds to list
  preds_list <- c(preds_list, preds)
  
  #puszczenie evaluacji modelu powoduje ze index predykcji przeksakuje o jeden batch_size do przodu
  #evaluate model
  
  print("evaluating...")
  eval <- saved_model %>% evaluate(
    Xtest,
    ytest
  )
  
  print(eval)
  
  
  eval_list <- c(eval_list, eval)
  
  
  #generate signal
  
  print("generating signals...")
  
  #print(preds)
  
  for (j in 1:length(preds)){
    #if(!is.nan(preds[j])) {
    #if (eval < -0.002) {
    if(preds[j] >0.000000) {
      signal <- c(signal,1)
    } else if (preds[j]<0.000000){
      signal <- c(signal,-1)
    } else {
      signal <- c(signal,0)
    }
    #} else {
    #  signal <- c(signal,1)
    #}
    #}
    
  }
  
  print(length(test))
  print(length(signal))
  
  
  #write_feather(as.data.frame(signal), here("EXPORTS/signal_snp_1d.feather"))
  
  #Sys.sleep(5)
  
  
}


print(eval_list)
#mean(eval_list)
#summary(eval_list)

#made dla buy and hold
made(as.numeric(tail(data, length(preds_list))),as.numeric(tail(data, length(preds_list))))
#made dla sieci
made(as.numeric(tail(data, length(preds_list))),preds_list)

#signal <- tail(signal, 3267)

#write_feather(as.data.frame(signal), here("signal_snp1d.feather"))


#replace signal values to test different strategies
signal_bkp <- signal
#signal <- signal_bkp

#save for export
rets_pr <- as.data.frame(preds_list)
rets_pr$signal <- signal



#LONG ONLY
signal_l <-vector()
for (j in 1:length(signal)){
  
  if(signal[j] == 1 ) {
    signal_l[j] <- 1
  } else if (signal[j] == -1){
    signal_l[j] <- 0
  } else {
    signal_l <- c(signal_l,0)
  }
  
}
rets_pr$signal_long <- signal_l





#signal <- rets_pr$signal
#snp 1d price data
dt_1d <-
  read.csv(here("DATA/NKX.csv")) %>%
  select(1,5)
#dla ar
#dt_1d <- tail(dt_1d,length(signal))
dt_1d <- dt_1d[(n_train+1):(n_train + length(signal)),]
dt_1d$signal <- signal
dt_1d$signal_l <- signal_l
dt_1d <-xts(dt_1d,as.Date(dt_1d$Date))
dt_1d$Date <- NULL
storage.mode(dt_1d) <- "numeric"
names(dt_1d) <- c("close","signal","signal_l")



#generate equity line values

eql1 <-
  dt_1d %>% 
  get_equity_line(prices_var = "close", 
                  signal_var = "signal")
eql1 %>% head()


eql2 <-
  dt_1d %>% 
  get_equity_line(prices_var = "close", 
                  signal_var = "signal_l")
eql2 %>% head()



#draw eqline with signal, same start
eql1 %>%
  dygraph() %>%
  dyRangeSelector(height = 40) %>%
  dySeries("signal", axis = "y2", strokeWidth=0) %>%
  dyAxis("y2", label = "signal", independentTicks = TRUE) %>%
  dyAxis("y", label = "eql")


#stats LS
st1 <-
  eql1 %>%
  .[, "close"] %>%
  as.numeric() %>%
  getPerformanceStats(scale = 252)
st2 <-
  eql1 %>%
  .[, "eql"] %>%
  as.numeric() %>%
  getPerformanceStats(scale = 252)
st <- bind_rows(st1, st2)
rownames(st) <- c("btc", "eql")
st %>%
  kableExtra::kable(
    caption = "Performance stats based on daily intervals",
    row.names = T, digits = 2, escape = F
  ) %>%
  kableExtra::kable_styling(
    bootstrap_options = c("striped", "hover", "condensed", "responsive" ),
    full_width = T, font_size = 14
  )



#draw eqline with signal, same start
eql2 %>%
  dygraph() %>%
  dyRangeSelector(height = 40) %>%
  dySeries("signal_l", axis = "y2", strokeWidth=0) %>%
  dyAxis("y2", label = "signal_l", independentTicks = TRUE) %>%
  dyAxis("y", label = "eql")


#stats LO
st1 <-
  eql2 %>%
  .[, "close"] %>%
  as.numeric() %>%
  getPerformanceStats(scale = 252)
st2 <-
  eql2 %>%
  .[, "eql"] %>%
  as.numeric() %>%
  getPerformanceStats(scale = 252)
st <- bind_rows(st1, st2)
rownames(st) <- c("btc", "eql")
st %>%
  kableExtra::kable(
    caption = "Performance stats based on daily intervals",
    row.names = T, digits = 2, escape = F
  ) %>%
  kableExtra::kable_styling(
    bootstrap_options = c("striped", "hover", "condensed", "responsive" ),
    full_width = T, font_size = 14
  )







rets_pr$eql_bh <- eql1$close
rets_pr$eql_ls <- eql1$eql

rets_pr$eql_lo <- eql2$eql




write_feather(as.data.frame(rets_pr), here("EXPORTS/rets_nkx_madl_lstm.feather"))



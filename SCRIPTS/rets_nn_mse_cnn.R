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
library(MLmetrics)

source(here("SCRIPTS/functions.R"))




#loading 1day snp data
dt_1d <-
  read.csv(here("DATA/BVP.csv"))

snp <-
  dt_1d %>%
  select(5) %>%
  mutate(Close = Close/xts::lag.xts(Close) - 1) #proste

#logarytmiczne
snp <- diff(log(dt_1d$Close))
#snp <-log((snp+1))

#transform to matrix and drop NA
data <- data.matrix(snp)
data <- na.omit(data)

summary(data)




#define network hyperparams
filters <- 256
kernel_size <- 2
pool_size = 2
hidden_dims <- 250
dropout_rate <- 0.0002
batch_size <- 756
l2_reg <- 0.00001
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
  
  
  #loss <- (-5. * tf$sign(y_true * y_pred) * (tf$square(y_true-y_pred)))
  
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
  opt = optimizer_adam(lr = 0.00015) #dla dense wyzsza
  
  
  #crate train set
  train <- matrix(data[((i-n_train)+1):i])
  
  print(length(train))
  
  #calculate max and min values and divide by greater of them (normalize)
  m <- if (abs(max(train))>abs(min(train))) abs(max(train)) else abs(min(train))
  train <- train/m
  
  #create test set and divide by train min/max value
  test <- (matrix(data[((i+1)-n_steps):((i+n_test))])) 
  test <- na.omit(test)
  test <- test/m
  
  
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
    

    ###CNN LSTM I CONVOLUTED LSTM TUTAJ https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    layer_conv_1d(
      filters,
      kernel_size,
      activation = "relu",
      input_shape = input_shape
    ) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_max_pooling_1d(pool_size = pool_size) %>%

    layer_flatten() %>%
    
    layer_dense(64) %>%
  
    layer_dense(1)
  
  #compile the model
  model %>% compile(
    loss = "mse",
    optimizer = opt,
    #metrics = "mae"
  )
  
  
  print("trainin the model...")
  
  
  hist_RNN <- model%>%
    fit(
      X,
      y,
      batch_size = batch_size,
      validation_split = 0.375,
      epochs = 230,
      verbose = 2,
      shuffle = FALSE, #org było bez tego
      callbacks = list(mc)
    )
  
  
  
  #load savel model from mc
  #since we added restore_best_weights = TRUE model = saved_model
  saved_model = load_model_hdf5(here('best_model.h5'), compile=FALSE)
  saved_model %>% compile(
    optimizer=opt,
    loss = "mse"
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
  
  ##add preds to list, reverse normalization
  preds_list <- c(preds_list, (preds*m))
  
  #puszczenie evaluacji modelu powoduje ze index predykcji przeksakuje o jeden batch_size do przodu
  #evaluate model
  
  #TU nie ewaluujemy, robimy to niżej dla całego zbioru
  # print("evaluating...")
  # eval <- saved_model %>% evaluate(
  #   Xtest,
  #   ytest
  # )
  # 
  # print(eval)
  # 
  # 
  # eval_list <- c(eval_list, eval)
  
  
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
  
  
  write_feather(as.data.frame(signal), here("EXPORTS/signal_snp_1d.feather"))
  
  Sys.sleep(5)
  
  
}


#print(eval_list)
#mean(eval_list)
#summary(eval_list)





act <- tail(data, length(preds_list))



#MSE
mean((act - (preds_list))^2)
#MAE
mean(abs(act - (preds_list)))
#MAPE
mean(abs((as.numeric(act)-preds_list)/as.numeric(act))) * 100


#convert to simple??
#act <- (exp(act) -1)
#made max
made(as.numeric(act),as.numeric(act))
#made dla sieci
made(as.numeric(act),preds_list)





plot(tail(data, length(preds_list)),type="l")
lines(preds_list, col="red")

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








#signal <- tail(signal,1000)


##ARMA
#read arma forecasts
# armaf <- arrow::read_feather(here('EXPORTS/arma_bvp_ar0.feather'))
# 
# tail(armaf)
# 
# tail(data)
# 
# #log((snp+1))
# 
# #made(log(armaf$test+1), log(armaf$preds+1))
# 
# #MSE(log(armaf$test+1), log(armaf$preds+1))
# 
# made(armaf$test, armaf$preds)
# 
# MSE(armaf$test, armaf$preds)
# 
# signal_ar <- vector()
# 
# for (j in 1:nrow(armaf)){
# 
#   if(armaf$preds[j] >0.000000) {
#     signal_ar <- c(signal_ar,1)
#   } else if (armaf$preds[j]<0.000000){
#     signal_ar <- c(signal_ar,-1)
#   } else {
#     signal_ar <- c(signal_ar,0)
#   }
# 
# }
# 
# 
# 
# ###PROBABILISTIC
# probf <- arrow::read_feather(here('EXPORTS/prob_df.feather'))
# 
# tail(probf)
# 
# tail(data)
# 
# made(armaf$test, armaf$preds)
# 
# signal_ar <- vector()
# 
# probf['0']
# 
# for (j in 1:nrow(probf)){
#   
#   if(probf['0'][[1]][j] >0.000000) {
#     signal_ar <- c(signal_ar,1)
#   } else if (probf['0'][[1]][j]<0.000000){
#     signal_ar <- c(signal_ar,-1)
#   } else {
#     signal_ar <- c(signal_ar,0)
#   }
#   
# }
# 
# 
# #double signal strength (sparować z ar)
# signal_c <-vector()
# for (j in 1:length(signal)){
#   print(signal[j])
#   print(signal_ar[j])
#   
#   if(signal[j] == 1 && signal_ar[j] == 1 ) {
#     
#     signal_c[j] <- 1
#   } else if (signal[j] == -1 && signal_ar[j] == -1){
#     signal_c[j] <- -1
#   } else {
#     signal_c[j] <- 0
#   }
#   
# }




#signal <- rets_pr$signal
#snp 1d price data
dt_1d <-
  read.csv(here("DATA/KOSPI.csv")) %>%
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





# saveRDS(eql1,here("EXPORTS/eql_snp_1d.rds"))
# 
# tb <- xts2tbl(eql1)
# 
# 
# p <-
#   tb %>%
#   select(-starts_with("signal")) %>%
#   pivot_longer(cols = c(close, eqlLS)) %>%
#   
#   ggplot(aes(x = datetime, y = value, col = name)) +
#   geom_line() +
#   theme_bw() 
# 
# p


rets_pr$eql_bh <- eql1$close
rets_pr$eql_ls <- eql1$eql

rets_pr$eql_lo <- eql2$eql

write_feather(as.data.frame(rets_pr), here("EXPORTS/rets_kospi_mse_cnn.feather"))



#preds_list <- preds_list*m


#act <- tail(data, length(preds_list))
#double signal strength (sparować z ar)
ok <-vector()
for (j in 1:length(preds_list)){
  
  if(sign(preds_list[j]) ==  sign(act[j]) ) {
    
    ok[j] <- 1
  } else {
    ok[j] <- 0
  } 
  
}

table(ok)

table(ok)[2]/length(preds_list)*100








#predykcja z danymi na jutro

data <- matrix(1:3002)

test <- data[(nrow(data)-666+1):nrow(data)]

#split test data into samples and targets
splt_t <- split_sequence(test,5)
Xtest <- splt_t[[1]]
ytest <- splt_t[[2]]

#reshape test data
Xtest <- array(Xtest, dim = c(dim(Xtest)[1], dim(Xtest)[2],n_features))


print("making predictions...")
#make predictions on test set

preds <- model_RNN %>% predict(
  Xtest,
  verbose = 2
)



#testy
signal <- vector()

for (i in 1:length(preds)){
  if(preds[i] >0) {
    signal <- c(signal,1)
  } else if (preds[i]<0){
    signal <- c(signal,-1)
  } else {
    signal <- c(signal,0)
  }
}

y_true = tf$constant(list(-0.05, -0.02, -0.03, 0.01, 0.02, 0.01))
y_pred = tf$constant(list(0.05, -0.02, -0.03, 0.01, 0.02, 0.01))

loss <- (-1 * tf$sign(y_true * y_pred) * (tf$abs(y_true)))

loss <- (-1 * tf$sign(y_true * y_pred) )

loss

tf$reduce_mean(loss)

tf$reduce_mean(tf$constant(list(-1,1,1,1,1,1)))

mean(c(-0.05, -0.02, -0.03, 0.01, 0.02, 0.01))

#test for data preparation
data <- matrix(1:4320)

n_train = 3000
n_test = 100
n_records = length(data)

for (i in seq(n_train, n_records, n_test)){
  
  X<-data
  
  #train set is equal to n_train - validation set
  train <- matrix(X[(i-n_train):(i-(n_train*0.2))])
  #ma <- if (abs(max(train))>abs(min(train))) max(train) else min(train)
  #train <- train/ma
  
  val <- (matrix(X[(i-(n_train*0.2)+1):(i)]))
  test <- (matrix(X[(i+1-5):(i+n_test)]))
  #test <- matrix(as.numeric(na.omit(test)))
  
  print(i)
  
  
  #print(train)
  #print(val)
  print(test)
  
}


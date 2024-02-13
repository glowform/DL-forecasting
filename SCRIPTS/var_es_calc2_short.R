### Do przeliczenia es 1%

Sys.setenv(TZ='America/New_York')

library(xts)
library(tidyverse)
library(lubridate)
library(rugarch)
library(feather)
library(here)
library(TTR)
library(scales)
library(dplyr)
library(psych)
library(forecast)

dane<-read.csv(here("DATA/KOSPI.csv"), sep = ",", dec=".", header = TRUE, 
               stringsAsFactors=FALSE, row.names = "Date")
#dane<- tail(dane,3000)

plot(dane$Close, type="l")

returns_pct<-as.matrix(diff(log(dane$Close))*100)


#Read GARCH Data
garch_forecasts_dt <- as.data.frame(read_feather(here("EXPORTS/garch_forecasts_dt_kospi.feather")))
#Read NN Data
lstm_sigma_dt <- arrow::read_feather(here('EXPORTS/lstm_sigma_dt_kospi.feather'))

nn_garch_f <- nrow(garch_forecasts_dt)
nn_forecats <- nrow(lstm_sigma_dt)

lstm_sigma <- tail(as.matrix(lstm_sigma_dt['LSTM-apARCH-NORM']),nn_forecats)
SvOLHC <- tail(as.matrix(garch_forecasts_dt['vGKYZ_252']),nn_forecats)
#nn_forecats <- length(gru_sigma)

g_model <- "apARCH"
dist <- "norm"

#ls_10 <- as.matrix((na.omit(ma_10(lstm_sigma))))
#nn_forecats <- nrow(ls_10) #1194

#trim to match NN forecasts
rets <- tail(returns_pct,nn_forecats)
garch_sigma <- tail(garch_forecasts_dt[,paste(g_model, dist, "sigma", sep='_')],nn_forecats)
garch_mu <- tail(garch_forecasts_dt[,paste(g_model, dist, "mu", sep='_')],nn_forecats)

skew = tail(garch_forecasts_dt[,paste(g_model, dist, "skew", sep='_')],nn_forecats)
shape = tail(garch_forecasts_dt[,paste(g_model, dist, "shape", sep='_')],nn_forecats)

#plot forecasted volatility
plot.ts(tail(SvOLHC,nn_forecats))
lines(lstm_sigma, col="purple")
lines(tail(garch_sigma,nn_forecats), col="red")



#calculate VaR
# location+scale invariance allows to use [mu + sigma*q(p,0,1,skew,shape)]
var_calc <- function(sigma,dist,p,skew,shape) {
  
  VaR <- garch_mu + sigma*qdist(distribution = dist, p, mu = 0, sigma = 1, skew = skew, shape = shape )
  return(VaR)
  
}

dist = dist
p = 0.01


VaR_garch <- var_calc(garch_sigma,dist,p,skew,shape)
print(VaRTest(p, as.numeric(rets), as.numeric(VaR_garch)))




VaR_lstm <- var_calc(lstm_sigma,dist,p,skew,shape)
print(VaRTest(p, as.numeric(rets), as.numeric(VaR_lstm)))




# calculate ES #https://rdrr.io/cran/rugarch/man/ESTest.html

#we need to calculate integrate value for each observation separately and store it in new vector
ig_val <- vector()
f <- function(x) qdist(dist, p=x, mu = 0, sigma = 1,  shape = ishape, skew = iskew,
)

for (i in 1:nn_forecats) {
  iskew = skew[i]
  ishape = shape[i]
  
  ig_val <- c(ig_val,integrate(f, 0, p)$value)  
  
}


ES_garch <- garch_mu + garch_sigma*ig_val/p
print(ESTest(p, as.numeric(rets), ES_garch, VaR_garch, boot = TRUE, n.boot = 1000))

ES_lstm <- garch_mu + lstm_sigma*ig_val/p
print(ESTest(p, as.numeric(rets), ES_lstm, VaR_lstm, boot = TRUE, n.boot = 1000))


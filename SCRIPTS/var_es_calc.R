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

dane<-read.csv(here("DATA/SPX.csv"), sep = ",", dec=".", header = TRUE, 
               stringsAsFactors=FALSE, row.names = "Date")
#dane<- tail(dane,3000)

plot(dane$Close, type="l")

returns_pct<-as.matrix(diff(log(dane$Close))*100)
returns<-as.matrix(diff(log(dane$Close)))
plot(returns, type="l")

dane$Date <- rownames(dane)





vClose <- unlist(volatility(dane, calc="close", N=260)*10) #mnożymy razy 10 w związku z PROCENTOWYMI stopami zwrotu
vClose1 <- unlist(volatility(dane, calc="close", N=1)*10*16) #możymy dodatkowo razy 16? - pierwiastek z 260


vClose0 <- unlist(volatility(dane, calc="close", mean0=TRUE, N=1,n=1))
vGK <- unlist(volatility(dane, calc="garman", N=1,n=1))
vParkinson <- unlist(volatility(dane, calc="parkinson", N=1,n=1))
vRS <- unlist(volatility(dane, calc="rogers", N=1,n=1))
vGKYZ <- unlist(volatility(dane, calc="gk.yz", N=1,n=1))*10*sqrt(252)
vYZ <- unlist(volatility(dane, calc="yang.zhang", N=1,n=2))

plot(vGKYZ, type="l")

vGKYZ_n10 <- unlist(volatility(dane, calc="gk.yz", N=1,n=10))*10*sqrt(252)


plot(vGKYZ_n10, type="l")

ma_10 <- function(x, n = 10){stats::filter(x, rep(1 / n, n), sides = 1)}

vGKYZ_n10_2 <- ma_10(as.numeric(vGKYZ))

plot(ma_10(vGKYZ), type="l")

plot(vGKYZ, type="l")
lines(vGKYZ_n10, col="red")
lines(vGKYZ_n10_2, col="green")


ggplot(data.frame(time = 1:length(vGKYZ), vol = vGKYZ),
       aes(x = time, y = vol)) +
  geom_line()

acf(returns_pct)
pacf(returns_pct)


#read prob data
probf <- arrow::read_feather(here('EXPORTS/prob_df.feather'))
nn_forecats <- nrow(probf)
garch_mu <- as.numeric(probf['0'][[1]])
garch_sigma <- as.numeric(probf$sigma)


#Read GARCH Data
garch_forecasts_dt <- as.data.frame(read_feather(here("EXPORTS/garch_forecasts_dt_snp.feather")))
#Read NN Data
lstm_sigma_dt <- arrow::read_feather(here('EXPORTS/lstm_sigma_dt_snp.feather'))

nn_garch_f <- nrow(garch_forecasts_dt)
nn_forecats <- nrow(lstm_sigma_dt)

lstm_sigma <- tail(as.matrix(lstm_sigma_dt['LSTM-GARCH-SSTD']),nn_forecats)
SvOLHC <- tail(as.matrix(garch_forecasts_dt['vGKYZ_252']),nn_forecats)
#nn_forecats <- length(gru_sigma)

g_model <- "sGARCH"
dist <- "sstd"

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
 
#calculate MSE
mean((as.numeric(tail(SvOLHC,nn_forecats)) - as.numeric(tail(garch_sigma,nn_forecats)))^2)
#for GRU
mean((as.numeric(tail(SvOLHC,nn_forecats)) - as.numeric(lstm_sigma))^2)

which(is.na(garch_sigma), arr.ind=TRUE)

e1 <- (as.numeric(tail(SvOLHC,nn_forecats)) - as.numeric(tail(garch_sigma,nn_forecats)))^2
e2 <- (as.numeric(tail(SvOLHC,nn_forecats)) - as.numeric(lstm_sigma))^2

dm.test(e1,e2,h=1, alternative = "greater")


#calculate MAE
mean(abs(as.numeric(tail(SvOLHC,nn_forecats)) - as.numeric(tail(garch_sigma,nn_forecats))))
#for GRU
mean(abs(as.numeric(tail(SvOLHC,nn_forecats)) - as.numeric(lstm_sigma)))

#calculate HMSE
mean((1 - (as.numeric(tail(garch_sigma,nn_forecats))/as.numeric(tail(SvOLHC,nn_forecats))))^2)
#for GRU
mean((1 - (as.numeric(tail(lstm_sigma,nn_forecats))/as.numeric(tail(SvOLHC,nn_forecats))))^2)


sv_tr = tail(SvOLHC,nn_forecats)
RV4 = sv_tr

#regresja MZ dla prognoz modelu GARCH

RV4_sq = RV4^2

#mean((as.numeric(tail(RV4,nn_forecats)) - as.numeric(garch_sigma))^2)

vol_forecasts <- garch_sigma^2

MZ_reg <- lm(RV4_sq~vol_forecasts)

summary(MZ_reg)

R2_mzgarch <- summary(MZ_reg)$r.squared
R2_mzgarch

#suma kwadratow reszt modelu garch
SSE1 <- sum(MZ_reg$residuals^2)

SSE0 <- sum((RV4_sq - vol_forecasts)^2)

Femp <- (SSE0-SSE1)/2*(nn_forecats-2)/SSE1

p_value_mzgarch <- 1- pf(Femp,2,nn_forecats-2) 
p_value_mzgarch

#mocno odrzucamy H0 - prognozy są mocno obciążone



#regresja MZ dla prognoz sieci GRU

RV4_sq = tail(RV4,nn_forecats)^2

vol_forecasts <- lstm_sigma^2

MZ_reg <- lm(RV4_sq~vol_forecasts)

summary(MZ_reg)

R2_mzlstm <- summary(MZ_reg)$r.squared
R2_mzlstm

#suma kwadratow reszt modelu gru
SSE1 <- sum(MZ_reg$residuals^2)

SSE0 <- sum((RV4_sq - vol_forecasts)^2)

Femp <- (SSE0-SSE1)/2*(nn_forecats-2)/SSE1

p_value_mzlstm <- 1- pf(Femp,2,nn_forecats-2) 
p_value_mzlstm




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



#exporting for python
#tutaj wybieramy najlepsze modele dla 5% i 1% z tabelki
VaR_exp <- as.data.frame(VaR_garch)
colnames(VaR_exp)[1] <- "VaR 5% - EGARCH-STD"
rownames(VaR_exp) <- as.Date(tail(dane$Date,nn_forecats))
VaR_exp$Date <- as.Date(tail(dane$Date,nn_forecats))

VaR_exp['VaR 1% - GJR-GARCH-STD'] <- as.data.frame(VaR_garch)

var_hits_lstm <- (rets-VaR_lstm)<0
var_hits_garch <- (rets-VaR_garch)<0

VaR_exp['VaR 5% hits'] <- var_hits_garch
VaR_exp['VaR 1% hits'] <- var_hits_garch

arrow::write_feather(VaR_exp,here('EXPORTS/VaR_bvp.feather'))







VaR_RV <- var_calc(tail(SvOLHC,nn_forecats),dist,p,skew,shape)
print(VaRTest(p, as.numeric(rets), as.numeric(VaR_RV)))

plot.ts(tail(vGKYZ,nn_forecats))
lines(garch_sigma, col="green")
#lines(tail(SvTR,nn_forecats), col="purple")

plot(rets^2, type="l")


#plotting with date
plot(VaR_garch ~ Date, VaR_garch, xaxt = "n", type = "l")
axis(1, VaR_garch$Date,format(VaR_garch$Date, "%b %y"), cex.axis = .7)
#or
ggplot( data = VaR_garch, aes( Date, VaR_garch )) + geom_line()




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


ES_RV <- garch_mu + tail(vGKYZ,nn_forecats)*ig_val/0.05
print(ESTest(0.05, as.numeric(rets), ES_RV, VaR_RV, boot = TRUE, n.boot = 1000))



plot(VaR_garch$VaR_garch, type="l")
lines(VaR_lstm, col="purple")
lines(VaR_RV, col="red")
lines(rets, col="green")


plot(ES_garch, type="l")
lines(ES_lstm, col="purple")
lines(ES_RV, col="red")
lines(rets, col="green")

#differences between ES
plot(ES_garch - ES_lstm, type="l")

#install.packages("psych")
library(psych)
describe(ES_garch - ES_lstm)

#ES_simple <- 100*(exp(ES/100)-1) #????
#print(ESTest(0.05, as.numeric(simple_rets), ES_simple, VaR_garch_simple, boot = TRUE))


plot(rets - VaR_lstm, type="l")

#przekroczenia
var_hits_lstm <- (rets-VaR_lstm)<0
var_hits_garch <- (rets-VaR_garch)<0

sum(var_hits_lstm)


plot(rets, type="l")
lines(-var_hits_lstm*5, type="p", col="red")
lines(-var_hits_garch*7, type="p", col="green")


#przekroczenia ES
es_hits_lstm <- (rets-ES_lstm)<0
es_hits_garch <- (rets-ES_garch)<0


sum(es_hits_garch)
sum(es_hits_lstm)


plot(rets, type="l")
lines(-es_hits_lstm*5, type="p", col="red")
lines(-es_hits_garch*7, type="p", col="green")

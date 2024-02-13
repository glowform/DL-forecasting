Sys.setenv(TZ='America/New_York')
#raczej ustawic with tz albo force tz

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
library(FinTS)

dane<-read.csv(here("DATA/SPX.csv"), sep = ",", dec=".", header = TRUE, 
               stringsAsFactors=FALSE, row.names = "Date")
#dane<- tail(dane,3000)

dane<- head(dane,(nrow(dane)-2495))

#head(dane)

tail(dane)

plot(dane$Close, type="l")

returns_pct<-as.matrix(diff(log(dane$Close))*100)
returns<-as.matrix(diff(log(dane$Close)))
plot(returns, type="l")



#sprawdzamy opóźnienia arma, zwracają też testy arch
n.cores <- 4                    #setting the number of cores for parallel computing

cl <- makePSOCKcluster(n.cores) #{parallel} needed!

arfit = autoarfima(data = returns_pct, ar.max = 4, ma.max = 2, 
                   criterion = c("BIC"), method = "full", include.mean = TRUE,
                   cluster = cl)

stopCluster(cl) 
arfit


#dodatkowe testy arch
Box.test(coredata(returns_pct^2), type = "Ljung-Box", lag = 6)



ArchTest(returns_pct, lags = 6 )




#nn_forecats <- 1194


vClose <- unlist(volatility(dane, calc="close", N=260)*10) #mnożymy razy 10 w związku z PROCENTOWYMI stopami zwrotu
vClose1 <- unlist(volatility(dane, calc="close", N=1)*10*16) #możymy dodatkowo razy 16? - pierwiastek z 260


vClose0 <- unlist(volatility(dane, calc="close", mean0=TRUE, N=1,n=1))
vGK <- unlist(volatility(dane, calc="garman", N=1,n=1))
vParkinson <- unlist(volatility(dane, calc="parkinson", N=1,n=1))
vRS <- unlist(volatility(dane, calc="rogers", N=1,n=1))
vGKYZ <- unlist(volatility(dane, calc="gk.yz", N=1,n=1))
vYZ <- unlist(volatility(dane, calc="yang.zhang", N=1,n=2))

acf(returns_pct)
pacf(returns_pct)

g_model <- c("sGARCH","gjrGARCH","eGARCH","apARCH")
dist <- c("norm","std","sstd")

#g_model <- c("apARCH")
#dist <- c("sstd")

for (j in g_model) {

  print(paste("working on model: ", j))

  for (i in dist) {
    print(paste("working on dist: ", i))

    GARCH_spec <- ugarchspec(variance.model = list(model = j, garchOrder = c(1,1)), 
                            mean.model = list(armaOrder = c(0,0), include.mean = TRUE, archm = FALSE, 
                                              archpow = 1, arfima = FALSE, external.regressors = NULL, archex = FALSE), 
                            distribution.model = i)
    n.cores <- 4                   #setting the number of cores for parallel computing
    cl <- makePSOCKcluster(n.cores) #{parallel} needed! 
    forecasts_roll <- ugarchroll(GARCH_spec, as.numeric(unlist(returns_pct)), n.ahead = 1, forecast.length = length(returns)-504, 
                                refit.every = 1, refit.window = "moving", window.size = 504,
                                calculate.VaR=TRUE, VaR.alpha=c(0.01, 0.05), solver='hybrid', #solver.control = list(tol = 1e-3),
                                cluster = cl) #ten solver.control tylko dla aparcha, ale w przypadku stop nieprocentowych trzeba go dodać

    stopCluster(cl)                 #turning off parallel computing

    #report(forecasts_roll, VaR.alpha=0.05)

    garch_forecasts <- as.data.frame(forecasts_roll, density)
    nn_garch_f <- nrow(garch_forecasts)

    if (j == "sGARCH" & i == "norm" ) {
      print("creating data frame")
      garch_forecasts_dt<- as.data.frame(garch_forecasts[,c("Realized")])
      colnames(garch_forecasts_dt)[1] <- "Realized Ret"
    }
    

    garch_forecasts_dt[,paste(j, i, "mu", sep='_')] <- garch_forecasts[,"Mu"]
    garch_forecasts_dt[,paste(j, i, "sigma", sep='_')] <- garch_forecasts[,"Sigma"]
    garch_forecasts_dt[,paste(j, i, "skew", sep='_')] <- garch_forecasts[,"Skew"]
    garch_forecasts_dt[,paste(j, i, "shape", sep='_')] <- garch_forecasts[,"Shape"]

    view(garch_forecasts_dt)

  }#end dist loop
}#end model loop


garch_forecasts_dt_bkp <- garch_forecasts_dt

#for snp only
#garch_forecasts_dt <- garch_forecasts_dt %>% relocate("eGARCH_sstd_mu","eGARCH_sstd_sigma","eGARCH_sstd_skew", "eGARCH_sstd_shape", .before = "apARCH_norm_mu")



garch_forecasts_dt_short <- as.data.frame(read_feather(here("EXPORTS/short/garch_forecasts_dt_dax.feather")))

#for snp only
#garch_forecasts_dt_short <- rename(garch_forecasts_dt_short, "sGARCH_std_shape" = "GARCH_std_shape" )


#select garch columns
garch_forecasts_dt_short <- garch_forecasts_dt_short %>% select(1:49)

#add previous exports
garch_forecasts_dt <- rbind(garch_forecasts_dt, garch_forecasts_dt_short)

garch_sigma <- as.matrix(garch_forecasts_dt['sGARCH_norm_sigma'])

nn_garch_f <- nrow(garch_sigma)


#garch_sigma <-  tail(forecasts_roll@forecast$density$Sigma,nn_forecats)

vOLHC <- vGKYZ
#trim to GARCH length
vOLHC_tr = tail(as.numeric(vOLHC),nn_garch_f)
rets_tr = tail(as.numeric(returns_pct),nn_garch_f)

#Fiszeder str 162 - skalowanie
a <- sqrt(mean(rets_tr^2))
b <- sqrt(mean(vOLHC_tr^2)) 

a/b
# a <- sum(abs(rets_tr))
# b <- sum(abs(vOLHC_tr))
SvOLHC = (a/b)*vOLHC_tr
#SvOLHC = (0.8)*vOLHC_tr



plot(vOLHC_tr*100, type="l")
lines(SvOLHC, col="red")
lines(vOLHC_tr*10*15, col="green")
lines(garch_sigma, col="purple")

#add vOL to dataset
garch_forecasts_dt$vYZ_252 <- tail(vYZ*10*sqrt(252),nn_garch_f)
garch_forecasts_dt$vGKYZ <- tail(vGKYZ*100,nn_garch_f)
garch_forecasts_dt$SvOLHC <- tail(SvOLHC,nn_garch_f)
garch_forecasts_dt$vGKYZ_252 <- tail(vGKYZ*10*sqrt(252),nn_garch_f)



#Export garch data
write_feather(garch_forecasts_dt, here('EXPORTS/garch_forecasts_dt_dax.feather'))


#Qucik VaR test
VaR_RV <- var_calc(tail(SvOLHC,nn_garch_f),"norm",0.05,skew,shape)
print(VaRTest(0.05, as.numeric(rets_tr), as.numeric(VaR_RV)))


garch_sigma <-  tail(forecasts_roll@forecast$density$Sigma,nn_forecats)
garch_mu <- tail(forecasts_roll@forecast$density$Mu,nn_forecats)
skew = tail(forecasts_roll@forecast$density$Skew,nn_forecats)
shape = tail(forecasts_roll@forecast$density$Shape,nn_forecats)


vGK <- unlist(volatility(dane, calc="garman", N=1,n=1))
vParkinson <- unlist(volatility(dane, calc="parkinson", N=1,n=1))
vRS <- unlist(volatility(dane, calc="rogers", N=1,n=1))
vGKYZ <- unlist(volatility(dane, calc="gk.yz", N=1,n=1))
vYZ <- unlist(volatility(dane, calc="yang.zhang", N=1,n=2))


#true range
#rownames(dane)<-1:nrow(dane)
#atr <- as.data.frame(ATR(dane[,c("High","Low","Close")], n=10))

vOLHC <- vGKYZ

#trim to GARCH length
vOLHC_tr = tail(as.numeric(vOLHC),nn_forecats)
rets_tr = tail(as.numeric(returns_pct),nn_forecats)


#Fiszeder str 162 - skalowanie
a <- sqrt(mean(rets_tr^2))
b <- sqrt(mean(vOLHC_tr^2)) 

a/b

# a <- sum(abs(rets_tr))
# b <- sum(abs(vOLHC_tr))
SvOLHC = (a/b)*vOLHC_tr
#SvOLHC = (0.8)*vOLHC_tr



#vOLHC_tr = tail(as.numeric(vRS),nn_forecats)

plot(garch_sigma, type="l")
lines(SvOLHC, col="red")
lines(vOLHC_tr*10*10, col="purple")
#lines(gru_sigma, col="green")

describe(SvOLHC)
describe(vOLHC_tr)
describe(garch_sigma)

lstm_df <- as.data.frame(read_feather(here("DATA/prob_df.feather")))


VaR <- garch_mu + vOLHC_tr*10*10 * qdist("sstd", p = 0.01, mu = 0, sigma = 1,
 skew = skew, shape = shape)

print(VaRTest(0.01, as.numeric(rets_tr), as.numeric(VaR)))

ig_val <- vector()
f <- function(x) qdist("sstd", p=x, mu = 0, sigma = 1,  shape = ishape, skew = iskew,
)

for (i in 1:nn_forecats) {
  iskew = skew[i]
  ishape = shape[i]
  
  ig_val <- c(ig_val,integrate(f, 0, 0.05)$value)  
  
}

ES<- garch_mu + SvOLHC*ig_val/0.05
print(ESTest(0.05, as.numeric(rets_tr), ES, VaR, boot = TRUE, n.boot = 1000))



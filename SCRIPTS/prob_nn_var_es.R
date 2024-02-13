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
library(scoringRules)
#library(scoringutils)


prob_forecasts <- arrow::read_feather(here('EXPORTS/prob_bvp_cnn2_sstd.feather'))


nn_forecats <- nrow(prob_forecasts)

loc <- prob_forecasts$`0`
#sigma <- prob_forecasts$sigma #dla norm

scale <- prob_forecasts$sigma
shape <- prob_forecasts$nu
skew <- prob_forecasts$xi
rets <- prob_forecasts$Realized_Ret


#garch 

prob_forecasts <- arrow::read_feather(here('EXPORTS/garch_forecasts_dt_kospi.feather'))

prob_forecasts <-tail(prob_forecasts, 2487)

mu <- prob_forecasts$apARCH_sstd_mu
sigma <- prob_forecasts$apARCH_sstd_sigma
shape <- prob_forecasts$apARCH_sstd_shape
skew <- prob_forecasts$apARCH_sstd_skew
rets <- prob_forecasts$`Realized Ret`

#mu <- loc #dla std i norm

nn_forecats <- nrow(prob_forecasts)

#do std
#sigma <- scale * sqrt(shape /(shape -2))


#do sstd z pipienia
#fi <- ((skew^2 - (1/(skew^2)))*2*shape*gamma(0.5*(shape+1)))/((skew + (1/skew))*(shape-1)*gamma(0.5*shape)*sqrt(pi*shape))
#gamm <- ((skew^3 + (1/(skew^3)))/(skew + (1/skew)))*(shape/(shape-2))

#mu <- loc + fi*scale #wartość oczekiwana
#sigma <- sqrt((gamm-fi^2)*(scale^2)) #odchylenie standardowe

which(is.na(sigma))


#plot(loc, type='l')
#lines(mu, col="green")

#mu dla sstd
#mu <- loc*(skew-(skew^-1))

#stdev dla sstd
#sigma <- sqrt((scale - (loc^2))*((skew^2)+(skew^-2))+(2*(loc^2)-scale))
#z domanów - zle
#mu <- loc + (gamma((shape-1)/2)*sqrt(shape-2))/(sqrt(pi)*gamma(shape/2)) *(skew-(1/skew))
#sigma <- sqrt(skew^2+skew^(-2)-1-m_dd^2)

 
#sigma <- scale
#mu <- loc



#CRPS i log
#z paczki - https://cran.r-project.org/web/packages/scoringRules/scoringRules.pdf

#mniejsze są lepsze
#unloadNamespace("scoringutils")
#mean(logs(y = rets, family = "normal", mean = mu, sd = sigma))

#mean(logs(y = rets, family = "t", location = loc, scale = scale, df = shape))

#mean(crps(y = rets, family = "normal", mean = mu, sd = sigma))

#mean(crps(y = rets, family = "t", location = loc, scale = scale, df = shape))



# byla wersja paczki 0.1.7.2, nowa jest 1.0 nie zwraca wyników testu

#require(remotes)
#require(devtools)
#install_version("scoringutils", version = "0.1.7.2", repos = "http://cran.us.r-project.org")

#lepsza? - https://cran.r-project.org/web/packages/scoringutils/scoringutils.pdf
#PIT jest też w rugarchu jako HLtest, ale 

library(scoringutils)

samp<- matrix(0, nrow = nn_forecats, ncol=100000)

for (i in 1:nn_forecats) {
  
  x <- rdist(distribution = "sstd", n=100000, mu = mu[i], sigma = sigma[i],
             shape=shape[i], 
             skew = skew[i]
             )
  dim(x) <- c(1, 100000)
 
  samp[i,] <- x

  print(i)
  
}

#mean(logs_sample(rets, samp))

#mean(crps_sample(rets, samp))

pit <- pit_sample(rets, samp)

mean(pit)



#plot_pit(pit)





###VaR
var_calc <- function(sigma,dist,p,skew,shape) {
  
  VaR <- mu + sigma*qdist(distribution = dist, p, mu = 0, sigma = 1, skew = skew, shape = shape )
  return(VaR)
  
}

dist = 'sstd'
p = 0.05


VaR_prob <- var_calc(sigma,dist,p,skew,shape)
print(VaRTest(p, as.numeric(rets), as.numeric(VaR_prob)))

which(is.na(VaR_prob))


plot(VaR_prob, type="l")
lines(VaR_q[,2], col="green")


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


ES_prob <- mu + sigma*ig_val/p
print(ESTest(p, as.numeric(rets), ES_prob, VaR_prob, boot = TRUE, n.boot = 1000))





#exporting for python
#tutaj wybieramy najlepsze modele dla 5% i 1% z tabelki


dane<-read.csv(here("DATA/WIG.csv"), sep = ",", dec=".", header = TRUE, 
               stringsAsFactors=FALSE, row.names = "Date")

VaR_exp <- as.data.frame(VaR_prob)
colnames(VaR_exp)[1] <- "VaR 5% - LSTM-STD"
rownames(VaR_exp) <- as.Date(tail(rownames(dane),nn_forecats))
VaR_exp$Date <- as.Date(tail(rownames(dane),nn_forecats))

var_hits_5 <- (rets-VaR_prob)<0
VaR_exp['VaR 5% hits'] <- var_hits_5


#VaR_exp <- arrow::read_feather(here('EXPORTS/VaR_bvp_prob.feather'))


VaR_exp['VaR 1% - CNN-STD'] <- as.data.frame(VaR_prob)

var_hits_1 <- (rets-VaR_prob)<0
VaR_exp['VaR 1% hits'] <- var_hits_1


arrow::write_feather(VaR_exp,here('EXPORTS/VaR_bvp_prob.feather'))


b=2487
c=100

a=8
a/b*c



###VaR próbkowy


probs <- c(0.05, 0.01)
VaR_q <- matrix(0, nrow = nn_forecats, ncol=2)

for (i in 1:nn_forecats) {
  
  x <- rdist(distribution = "sstd", n=100000, mu = mu[i], sigma = sigma[i], skew = skew[i], shape = shape[i])
  
  VaR_q[i,1] <- quantile(x, probs = 0.05) 
  VaR_q[i,2] <- quantile(x, probs = 0.01) 
  
  print(i)
  
}

#VaR_prob <- var_calc(sigma,dist,p,skew,shape)
print(VaRTest(p, as.numeric(rets), as.numeric(VaR_q[,2])))

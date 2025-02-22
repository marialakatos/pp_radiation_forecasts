library(scoringRules)
library(crch)

####### Calculates CN0 EMOS verification measures for WRF forecasts

rm(list=ls())
gc()

##### Variables to be chosen

### Length of the rolling training period
train <- 85

### Number of clusters
no.clust <- 6

### Minimum number of elements in a cluster
c.min <- 3 

### Specification of the confidence level of the central prediction interval (1 - 2*alpha)
alpha <- 1/9

##### Initialization #####

# Path for data/results and scripts
pathData <- ''      #Path should be modified according to the actual location of data/results

### Loading the data used for post-processing
load(paste(pathData,'radSumData.RData',sep = ""))

### Selecting observation dates, times and locations. 
commonDates <- dimnames(ensSumDay1)[[3]]
commonDatesN <- as.numeric(length(commonDates))

commonTimes <- dimnames(ensSumDay1)[[2]]
commonTimesN <- as.numeric(length(commonTimes))

commonLocs <- dimnames(ensSumDay1)[[1]]
commonLocsN <- as.numeric(length(commonLocs))

### Loading the parameters of the EMOS models. Output of EmosFitRadCN.R
load(paste(pathData,'radParsEmos_train',train,'_Clust',no.clust,'_CN',c.min,'.RData',sep = ''))


###Verification period (starting date: training period length + 3 days)
verDates <- dimnames(radParsEmos24)[[3]]  

### Replacement of 0 variances by a small value
tmp <- ensSumDay1[,,,3]
tmp[tmp<1e-20] <- 1e-20
ensSumDay1[,,,3] <- tmp 

tmp <- ensSumDay2[,,,3]
tmp[tmp<1e-20] <- 1e-20
ensSumDay2[,,,3] <- tmp 


### Calculated verification measures: 
# CRPS 
# Absolute error of median 
# Squared error of mean 
# Probability integral transform
# Coverage: 1 if the observation is in the central prediction interval, 0 otherwise
# Width of the central prediction interval
# Mean of the CN0 distribution
# Median of the CN0 distribution
# Location parameter of the CN0 distribution
# Scale parameter of the CN0 distribution

scoreNames <- c('CRPS','AE','SE','PIT','IsIn','Width','Mean','Median','LOC','SCALE')

# Separate storing of scores for 0-24h and 24-48h forecasts
# 4 dimensional arrays. 1: score; 2: date of validity of the forecast; 3: time of validity of the forecast; 4: location.
radScoresEmos24 <- radScoresEmos48 <- array(data = NA, dim = c(length(scoreNames),length(verDates),commonLocsN,commonTimesN), dimnames = list(scoreNames,verDates,commonLocs,commonTimes))

start.day <- train + 3 
end.day <- commonDatesN

##### Calculation of scores #####

### Calculating scores for each day of the verification period
for (d in c((start.day):(end.day))){
  act.date <- commonDates[d]          #Actual date
  print(act.date)
  obs.test <- verobs[,,act.date]   #Observations for the actual date
  ens.test24 <- ensSumDay1[,,act.date,] #Corresponding 0-24h ensemble features
  ens.test48 <- ensSumDay2[,,act.date,] #Corresponding 24-48h ensemble features
  emos.pars24 <- radParsEmos24[,,act.date,] #Corresponding 0-24h EMOS parameters
  emos.pars48 <- radParsEmos48[,,act.date,] #Corresponding 24-48h EMOS parameters
  LOC24 <- emos.pars24[,,1] + emos.pars24[,,2] * ens.test24[,,1] + emos.pars24[,,3] * ens.test24[,,2] #CN0 location, 0-24h
  LOC48 <- emos.pars48[,,1] + emos.pars48[,,2] * ens.test48[,,1] + emos.pars48[,,3] * ens.test48[,,2] #CN0 location, 24-48h
  SCALE24 <- exp(emos.pars24[,,4] + emos.pars24[,,5] * log(sqrt(ens.test24[,,3]))) #CL0 scale, 0-24h
  SCALE48 <- exp(emos.pars48[,,4] + emos.pars48[,,5] * log(sqrt(ens.test48[,,3]))) #CL0 scale, 24-48h
  
  err24 <- try(CRPS24 <- crps_cnorm(obs.test, location = LOC24, scale = SCALE24, lower = 0), silent =T) #CRPS, 0-24h
  # Handling the case of NA CRPS values
  if ((class(err24)[1] == "try-error") | all(is.na(CRPS24))){
    CRPS24 <- matrix(NA, ncol = commonTimesN, nrow = commonLocsN)
    for (loc in c(1:commonLocsN)){
      good.obs <- complete.cases(obs.test[loc,]) & complete.cases(LOC24[loc,])
      CRPS24[loc,good.obs] <- crps_cnorm(obs.test[loc,good.obs], location = LOC24[loc,good.obs], scale = SCALE24[loc,good.obs], lower = 0)
      
    }
    dimnames(CRPS24) <- list(commonLocs,commonTimes)
  }
  CRPS24[is.na(CRPS24)] <- NA
  
  err48 <- try(CRPS48 <- crps_cnorm(obs.test, location = LOC48, scale = SCALE48, lower = 0), silent = T) #CRPS, 24-48h
  # Handling the case of NA CRPS values
  if ((class(err48)[1] == "try-error") | all(is.na(CRPS48))){  
    CRPS48 <- matrix(NA, ncol = commonTimesN, nrow = commonLocsN)
    for (loc in c(1:commonLocsN)){
      good.obs <- complete.cases(obs.test[loc,]) & complete.cases(LOC48[loc,])
      CRPS48[loc,good.obs] <- crps_cnorm(obs.test[loc,good.obs], location = LOC48[loc,good.obs], scale = SCALE48[loc,good.obs], lower = 0)
    }
    dimnames(CRPS48) <- list(commonLocs,commonTimes)
  }
  
  CRPS48[is.na(CRPS48)] <- NA
  
  PIT24 <- pcnorm(obs.test, mean = LOC24, sd = SCALE24, left = 0) #PIT, 0-24h
  # Randomization for 0 observations
  zeroObs <- which(obs.test == 0, arr.ind = T)
  if (length(zeroObs)>0){
    zeroProb <- pcnorm(0, mean = LOC24[zeroObs], sd = SCALE24[zeroObs], left = 0)
    genPIT <- sapply(zeroProb, function(p){runif(1, min = 0, max = p)})
    PIT24[zeroObs] <- genPIT
  }
  
  PIT48 <- pcnorm(obs.test, mean = LOC48, sd = SCALE48, left = 0) #PIT, 24-48h
  # Randomization for 0 observations
  zeroObs <- which(obs.test == 0, arr.ind = T)
  if (length(zeroObs)>0){
    zeroProb <- pcnorm(0, mean = LOC48[zeroObs], sd = SCALE48[zeroObs], left = 0)
    genPIT <- sapply(zeroProb, function(p){runif(1, min = 0, max = p)})
    PIT48[zeroObs] <- genPIT
  }
  
  # Mean of CN0 distribution, 0-24h
  Z24 <- LOC24/SCALE24
  MEAN24 <- pnorm(Z24)*LOC24 + SCALE24 * dnorm(Z24)
  MEAN24[is.infinite(MEAN24)] <- 0
  # Mean of CN0 distribution, 24-48h
  Z48 <- LOC48/SCALE48
  MEAN48 <- pnorm(Z48)*LOC48 + SCALE48 * dnorm(Z48)
  MEAN48[is.infinite(MEAN48)] <- 0
  
  
  # Median of CN0 distribution, 0-24h
  MEDIAN24 <- qcnorm(0.5, mean = LOC24, sd = SCALE24, left = 0)
  # Median of CN0 distribution, 24-48h
  MEDIAN48 <- qcnorm(0.5, mean = LOC48, sd = SCALE48, left = 0)  
  
 
  # Evaluation of scores, 0-24h
  radScoresEmos24[1,act.date,,] <- CRPS24 #CRPS
  radScoresEmos24[2,act.date,,] <- abs(MEDIAN24 - obs.test) #AE of median
  radScoresEmos24[3,act.date,,] <- (MEAN24 - obs.test)^2 #SE of mean
  radScoresEmos24[4,act.date,,] <- PIT24 #PIT
  radScoresEmos24[5,act.date,,] <- (qcnorm(alpha, mean = LOC24, sd = SCALE24, left = 0) <= obs.test) & (qcnorm(1-alpha, mean = LOC24, sd = SCALE24, left = 0) >= obs.test) #Coverage
  radScoresEmos24[6,act.date,,] <- qcnorm(1-alpha, mean = LOC24, sd = SCALE24, left = 0) - qcnorm(alpha, mean = LOC24, sd = SCALE24, left = 0) #Width
  radScoresEmos24[7,act.date,,] <- MEAN24   #Mean
  radScoresEmos24[8,act.date,,] <- MEDIAN24 #Median
  radScoresEmos24[9,act.date,,] <- LOC24 #Location
  radScoresEmos24[10,act.date,,] <- SCALE24 #Scale

  # Evaluation of scores, 24-48h
  radScoresEmos48[1,act.date,,] <- CRPS48 #CRPS
  radScoresEmos48[2,act.date,,] <- abs(MEDIAN48 - obs.test) #AE of median
  radScoresEmos48[3,act.date,,] <- (MEAN48 - obs.test)^2 #SE of mean
  radScoresEmos48[4,act.date,,] <- PIT48 #PIT
  radScoresEmos48[5,act.date,,] <- (qcnorm(alpha, mean = LOC48, sd = SCALE48, left = 0) <= obs.test) & (qcnorm(1-alpha, mean = LOC48, sd = SCALE48, left = 0) >= obs.test) #Coverage
  radScoresEmos48[6,act.date,,] <- qcnorm(1-alpha, mean = LOC48, sd = SCALE48, left = 0) - qcnorm(alpha, mean = LOC48, sd = SCALE48, left = 0) #Width
  radScoresEmos48[7,act.date,,] <- MEAN48   #Mean
  radScoresEmos48[8,act.date,,] <- MEDIAN48 #Median
  radScoresEmos48[9,act.date,,] <- LOC48 #Location
  radScoresEmos48[10,act.date,,] <- SCALE48 #Scale

}

radScoresEmos24 <- aperm(radScoresEmos24,c(3,4,2,1))
radScoresEmos48 <- aperm(radScoresEmos48,c(3,4,2,1))

##### Saving the results #####
save(radScoresEmos24, radScoresEmos48, file = paste(pathData,'radScoresEmos_train',train,'_Clust',no.clust,'_CN',c.min,'.RData',sep = ''))


na_percentage_score24 <- mean(is.na(radScoresEmos24)) * 100
na_percentage_score48 <- mean(is.na(radScoresEmos48)) * 100

print(paste("Percentage of NA values in na_percentage_score24:", na_percentage_score24))
print(paste("Percentage of NA values in na_percentage_score48:", na_percentage_score48))

cat("MEAN CRPS (first 24 hours):", mean(radScoresEmos24[,,,'CRPS'], na.rm = T))
cat("MEAN CRPS (last 24 hours):", mean(radScoresEmos48[,,,'CRPS'], na.rm = T))

crps_per_lt2 <- c(apply(radScoresEmos24[,,,'CRPS'], 2, mean, na.rm = T), apply(radScoresEmos48[,,,'CRPS'], 2, mean, na.rm = T))
plot(crps_per_lt2, type = 'b')


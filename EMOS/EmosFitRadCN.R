library(abind)
library(foreach)
library(doParallel)
library(scoringRules)

####### Fits clustering based CN0 EMOS models to WRF forecasts and observations

# Ensemble members are considered exchangeable, 5 parameters

rm(list=ls())
gc()

##### Variables to be chosen

### Length of the rolling training period
train <- 85

### Number of clusters and features
no.clust <- 6
no.feat <- 24

### Minimum number of elements in a cluster
c.min <- 3 

##### Initialization #####

# Path for data/results and scripts

pathData <- '' #Path should be modified according to the actual location of data/results
pathScripts <- ''  #Path should be modified according to the actual location of scripts

### Loading the data for post-processing
# verobs: verifying observations; 3 dimensional array. 
# Dimensions: 1: location; 2: time of observation (1h time steps); 3: date of observation
# ensSumDay1: forecasts for 0-24h; 4 dimensional array. 
# ensSumDay2: forecasts for 24-48h; 4 dimensional array. 
# Dimensions: 1: location; 2: forecast horizon (1h time steps); 3: date of initialization; 4: feature (ensemble mean,proportion of 0 forecasts, ensemble variance)

load(paste(pathData,'radSumData.RData',sep = ""))

### Loading the functions necessary for modeling (fit.emos.reg.CN, acomb3, cluster.fit) 
source(paste(pathScripts,'EmosFuncRad.R',sep = '')) 

### Selecting observation dates, times and locations. 
commonDates <- dimnames(ensSumDay1)[[3]]
commonDatesN <- as.numeric(length(commonDates))

commonTimes <- dimnames(ensSumDay1)[[2]]
commonTimesN <- as.numeric(length(commonTimes))

commonLocs <- dimnames(ensSumDay1)[[1]]
commonLocsN <- as.numeric(length(commonLocs))


### Verification period (starting date: training period length + 3 days)
verDates <- commonDates[-(1:(train+2))]
start.day <- train + 3
end.day <- commonDatesN

### Parameters of the EMOS model. 
# Location: a (constant), bM (ensemble mean), b0 (proportion of 0 forecasts)
# Scale: c (constant), d (log of the ensemble variance)
parNames <- c('a','bM','b0', 'c','d')
parNamesN <- as.numeric(length(parNames))

# Separate storing of parameters for 0-24h and 24-48h forecasts
# 4 dimensional arrays. 1: parameter; 2: date of validity of the forecast; 3: time of validity of the forecast; 4: location
radParsEmos24 <- radParsEmos48 <- array(data = NA, dim = c(parNamesN,length(verDates),commonTimesN,commonLocsN), dimnames = list(parNames,verDates,commonTimes,commonLocs))


##### Modeling #####

### Separate model for each forecast horizon. 

cores <- detectCores()        #Detects the number of CPU cores
c1 <- makeCluster(cores[1]-2) #2 cores are left free not to overload the computer
registerDoParallel(c1)        #Registering the cluster for parallel computation

### Modeling for each day of the verification period
for (d in c((start.day):(end.day))){
  print(commonDates[d])
  start.time <- Sys.time()
  obs.train24 <- verobs[,,commonDates[(d-train-1):(d-2)]]       #Training observations for 0-24h forecasts. Mind the day shift
  obs.train48 <- verobs[,,commonDates[(d-train-2):(d-3)]]       #Training observations for 24-48h forecasts. Mind the day shift
  ens.train24 <- ensSumDay1[,,commonDates[(d-train-1):(d-2)],]  #Training ensemble features for 0-24h forecasts. Mind the day shift
  ens.train48 <- ensSumDay2[,,commonDates[(d-train-2):(d-3)],]  #Training ensemble features for 24-48h forecasts Mind the day shift
  
  ### Parallel modeling of the different forecast horizons, 0-24h. Output: EMOS parameters for each forecast horizon
  parsEmos24 <- foreach(tt = (1:commonTimesN),.combine='acomb3', .multicombine=TRUE, .packages = c('scoringRules')) %dopar% {
    act.obs.train24 <- obs.train24[,tt,]  #Training observations for the given horizon
    act.ens.train24 <- ens.train24[,tt,,] #Training ensemble features for the given horizon
    fit.pars <- matrix(NA, ncol = commonLocsN, nrow = parNamesN)
    
    # Creating clusters of stations based on training data
    cluster.id <- cluster.fit(act.obs.train24,act.ens.train24[,,1],commonLocsN,no.clust,method="both")
    
    no.clust.new <- no.clust
    
    # Ensuring to have at least c.min stations in each cluster
    while (min(table(cluster.id)) < c.min){
      no.clust.new <-no.clust.new - 1
      cluster.id <- cluster.fit(act.obs.train24,act.ens.train24[,,1],commonLocsN,no.clust.new,method="both")
    }
    
    # Estimating EMOS parameters for each cluster separately 
    for (i in c(1:no.clust.new)) {
      cluster.in <- which(cluster.id == i)
      loc.obs.train <- act.obs.train24[cluster.in,]
      fit.pars[,cluster.in] <- fit.emos.reg.CN(obs.train = loc.obs.train, ens.train = act.ens.train24[cluster.in,,]) #Estimation of model parameters
      
    }
    fit.pars
  }
  
 radParsEmos24[,d-start.day+1,,] <- aperm(parsEmos24,c(1,3,2))
  
  ### Parallel modeling of the different forecast horizons, 24-48h. Output: EMOS parameters for each forecast horizon
  parsEmos48 <- foreach(tt = (1:commonTimesN),.combine='acomb3', .multicombine=TRUE, .packages = c('scoringRules')) %dopar% {
    act.obs.train48 <- obs.train48[,tt,]  #Training observations for the given horizon
    act.ens.train48 <- ens.train48[,tt,,] #Training ensemble features for the given horizon
    fit.pars <- matrix(NA, ncol = commonLocsN, nrow = parNamesN)
    
    # Creating clusters of stations based on training data
    cluster.id <- cluster.fit(act.obs.train48,act.ens.train48[,,1],commonLocsN,no.clust,method="both")
   
    no.clust.new <- no.clust
    
    # Ensuring to have at least c.min stations in each cluster
    while (min(table(cluster.id)) < c.min){
      no.clust.new <-no.clust.new - 1
      cluster.id <- cluster.fit(act.obs.train48,act.ens.train48[,,1],commonLocsN,no.clust.new,method="both")
    }

    # Estimating EMOS parameters for each cluster separately 
    for (i in c(1:no.clust.new)) {
      cluster.in <- which(cluster.id == i)
      loc.obs.train <- act.obs.train48[cluster.in,]
      fit.pars[,cluster.in] <- fit.emos.reg.CN(obs.train = loc.obs.train, ens.train = act.ens.train48[cluster.in,,]) #Estimation of model parameters
    }
    fit.pars
  }
  
  radParsEmos48[,d-start.day+1,,] <- aperm(parsEmos48,c(1,3,2))
  end.time <- Sys.time()
  print(end.time-start.time)  
  
}

stopCluster(c1) #Stopping the cluster for parallel computation

radParsEmos24 <- aperm(radParsEmos24,c(4,3,2,1))
radParsEmos48 <- aperm(radParsEmos48,c(4,3,2,1))

##### Saving the results #####
save(radParsEmos24, radParsEmos48, file = paste(pathData,'radParsEmos_train',train,'_Clust',no.clust,'_CN',c.min,'.RData',sep = '')) 

### Contains all required functions for censored normal EMOS modeling.

  # fit.emos.reg.CN: estimates the parameters of the regional CN EMOS model, 5 parameters

  # acomb3: auxiliary function for parallel computing. Binds three dimensional arrays along the third dimension

  # cluster.fit: performs clustering of the observation stations


fit.emos.reg.CN <- function(obs.train, ens.train) {
  
  #Estimates the parameters of the regional CN0 EMOS model
  
  #Input: obs.train - observations in the training dataset. Array of size L x N
  #       ens.train - features extracted from the forecasts of the training dataset. Array of size L x N X 3
  #       ensemble mean (mean); proportion of 0 forecasts (p0); ensemble variance (var)
  #       L: number of locations
  #       N: length of the training period 
 
  
  
  #Output: parameters of the regional CN0 EMOS model
  #Order of parameters: a, bM, b0, c, d
  
  
  # Calculation of the initial values for the optimization. Regression of the observations on forecasts and (0,1) for (c,d)
  olsCoefs <-  lm(as.vector(obs.train) ~ as.vector(ens.train[,,1]) + as.vector(ens.train[,,2]))$coef
  olsCoefs[is.na(olsCoefs)] <- 0
  pars.start <- as.numeric(c(olsCoefs,0,1))
  ens.train[,,3][ens.train[,,3]<1e-20] <- 1e-20 # Replacement of 0 variances by a small value
  
  # Sum of the CRPS of the CN0 distribution over the training data as function of the parameters
  crpsCN <- function(pars){
    LOC <- as.vector(pars[1] + pars[2] * ens.train[,,1] + pars[3] * ens.train[,,2])
    SCALE <- as.vector(exp(pars[4] + pars[5] * log(sqrt(ens.train[,,3]))))
    OBS <- as.vector(obs.train)
    goodVals <- complete.cases(LOC) & complete.cases(OBS) & complete.cases(SCALE)
    CRPS <- crps_cnorm(y = OBS[goodVals], location = LOC[goodVals], scale = SCALE[goodVals], lower = 0)
    return(sum(CRPS, na.rm = T))
  }
  
  # Gradient of the CRPS of the CN0 distribution over the training data as function of the parameters
  grad.crpsCN <- function(pars){
    LOC <- as.vector(pars[1] + pars[2] * ens.train[,,1] + pars[3] * ens.train[,,2])
    SCALE <- as.vector(exp(pars[4] + pars[5] * log(sqrt(ens.train[,,3]))))
    OBS <- as.vector(obs.train)
    goodVals <- complete.cases(LOC) & complete.cases(OBS) & complete.cases(SCALE)
    grad.CRPS <- gradcrps_cnorm(y = OBS[goodVals], location = LOC[goodVals], scale = SCALE[goodVals], lower = 0)
    grad.res <- vector(length = length(pars))
    grad.res[1] <- sum(grad.CRPS[,'dloc'], na.rm = T)
    grad.res[2] <- sum(grad.CRPS[,'dloc'] * as.vector(ens.train[,,1])[goodVals], na.rm = T)
    grad.res[3] <- sum(grad.CRPS[,'dloc'] * as.vector(ens.train[,,2])[goodVals], na.rm = T)
    grad.res[4] <- sum(grad.CRPS[,'dscale'] * SCALE[goodVals], na.rm = T)
    grad.res[5] <- sum(grad.CRPS[,'dscale'] * SCALE[goodVals] * log(sqrt(as.vector(ens.train[,,3])[goodVals])), na.rm = T)
    return(grad.res)
  }
  
  # Optimization of the CRPS over the training data
  err <- try(opt <- optim(par = pars.start, fn = crpsCN, gr = grad.crpsCN, method="BFGS", control = list(maxit = 200))$par, silent = T)
  # Handling problematic initial values
  if (class(err) == "try-error"){
    opt <- optim(par = c(1,0,0,0,1), fn = crpsCN, gr = grad.crpsCN, method="BFGS", control = list(maxit = 200))$par
  }
  
  # Re-estimation of the unrealistic coefficients
  if (sum(abs(opt))>2000){
    err2 <- try(opt <- optim(par = pars.start, fn = crpsCN, gr = grad.crpsCN, method="L-BFGS-B", control = list(maxit = 200))$par, silent = T)
      if (class(err2) == "try-error"){
      opt <- optim(par = c(1,0,0,0,1), fn = crpsCN, method="L-BFGS-B", control = list(maxit = 200))$par
      }
     }
  
  return(opt)
  
}


acomb3 <- function(...) abind(..., along=3)


cluster.fit <- function(obs.train,ens.train.mean,nStat,no.clus,method='both') {
  switch (method,
          climat = {
            cluster.data <- apply(obs.train,1,quantile,probs=c(1:no.feat)/(no.feat+1),na.rm=T)
          },
          error = {
            cluster.data <- apply((ens.train.mean - obs.train),1,quantile,probs=c(1:no.feat)/(no.feat+1),na.rm=T)
          },
          forc = {
            cluster.data <- apply(ens.train.mean,1,quantile,probs=c(1:no.feat)/(no.feat+1),na.rm=T)
          },
          both = {
            n1 <- floor(no.feat/2)
            n2 <- no.feat - n1
            cluster.data <- rbind(apply(obs.train,1,quantile,probs=c(1:n1)/(n1+1),na.rm=T), apply((ens.train.mean - obs.train),1,quantile,probs=c(1:n2)/(n2+1),na.rm=T))
          }
  )		
  good.stat <- !apply(cluster.data,2,anyNA)
  cluster.id <- rep(NA,nStat)
  cluster.id[good.stat] <- kmeans(t(cluster.data[,good.stat]),no.clus,iter.max=20)$cluster		
  return(cluster.id)
}

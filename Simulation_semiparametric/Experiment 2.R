source('bridges.R')
###### Experiment 2: q0, q1, and q2 are misspecified #####

set.seed(2025)
library(MASS)
n <- 1000
nsimu <- 500

psi_POR <- rep(NA, nsimu)
psi_PIPW <- rep(NA, nsimu)
psi_Ph1 <- rep(NA, nsimu)
psi_Ph2 <- rep(NA, nsimu)
psi_PQR <- rep(NA, nsimu)

nboots <- 100
psiboots_POR <- rep(NA, nboots)
psiboots_PIPW <- rep(NA, nboots)
psiboots_Ph1 <- rep(NA, nboots)
psiboots_Ph2 <- rep(NA, nboots)
psiboots_PQR <- rep(NA, nboots)

l.POR <- rep(NA, nsimu)
u.POR <- rep(NA, nsimu)
l.PIPW <- rep(NA, nsimu)
u.PIPW <- rep(NA, nsimu)
l.Ph1 <- rep(NA, nsimu)
u.Ph1 <- rep(NA, nsimu)
l.Ph2 <- rep(NA, nsimu)
u.Ph2 <- rep(NA, nsimu)
l.PQR <- rep(NA, nsimu)
u.PQR <- rep(NA, nsimu)

# create a bootstrap function
fun_boots <- function(j) {
  index <- sample(1:n, n, replace = TRUE)
  XX <- X[index, ]
  AA <- A[index]
  ZZ <- Z[index]
  WW <- W[index]
  MM <- M[index]
  DD <- D[index]
  YY <- Y[index]
  data <- list(Z=ZZ, W=WW, A=AA, M=MM, D=DD, X=XX, Y=YY) # observed data: ZZ, WW, AA, MM, DD, XX, YY
  
  ##### proximal outcome regression ####
  inioptim <- c(0,0,0,0,0,0)
  h2par <- optim(par=inioptim, 
                 fn=bridgeh2, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  inioptim <- c(0,0,0,0,0)
  h1par <- optim(par=inioptim,
                 fn=bridgeh1, parah2=h2par, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  inioptim <- c(0,0,0,0)
  h0par <- optim(par=inioptim,
                 fn=bridgeh0, parah1=h1par, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  
  psiboots_POR[j] <<- mean( cbind(1,WW,XX) %*% h0par )
  
  #### proximal IPW ####
  inioptim <- c(0,0,0,0)
  q0par <- optim(par=inioptim,
                 fn=bridgeq0_mis, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  inioptim <- c(0,0,0,0,0)
  q1par <- optim(par=inioptim,
                 fn=bridgeq1_mis, paraq0=q0par, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  inioptim <- c(0,0,0,0,0,0)
  q2par <- optim(par=inioptim,
                 fn=bridgeq2_mis, paraq0=q0par, paraq1=q1par, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  
  psiboots_PIPW[j] <<- mean( AA*YY * ( 1+ exp(- cbind(1,ZZ,sqrt(abs(XX))+3 )%*%q0par ) ) * exp( cbind(1,ZZ,DD,sqrt(abs(XX))+3 )%*%q1par ) * exp( cbind(1,ZZ,MM,DD,sqrt(abs(XX))+3 )%*%q2par ) )
  
  #### proximal hybrid 1 ####
  psiboots_Ph1[j] <<- mean( AA * (cbind(1,WW,DD,XX)%*%h1par) * ( 1+ exp(- cbind(1,ZZ,sqrt(abs(XX))+3 )%*%q0par ) ) )
  
  #### proximal hybrid 2 ####
  psiboots_Ph2[j] <<- mean( (1-AA) * (cbind(1,WW,MM,DD,XX)%*%h2par) * ( 1+ exp(- cbind(1,ZZ,sqrt(abs(XX))+3 )%*%q0par ) ) * exp( cbind(1,ZZ,DD,sqrt(abs(XX))+3 )%*%q1par ) )
  
  #### proximal quadruply robust ####
  psiboots_PQR[j] <<- mean( AA * ( 1+ exp(- cbind(1,ZZ,sqrt(abs(XX))+3 )%*%q0par ) ) * ( cbind(1,WW,DD,XX)%*%h1par - cbind(1,WW,XX) %*% h0par ) 
                            + (1-AA) * ( 1+ exp(- cbind(1,ZZ,sqrt(abs(XX))+3 )%*%q0par ) ) * exp( cbind(1,ZZ,DD,sqrt(abs(XX))+3 )%*%q1par ) * ( cbind(1,WW,MM,DD,XX)%*%h2par - cbind(1,WW,DD,XX)%*%h1par )
                            + AA * ( 1+ exp(- cbind(1,ZZ,sqrt(abs(XX))+3 )%*%q0par ) ) * exp( cbind(1,ZZ,DD,sqrt(abs(XX))+3 )%*%q1par ) * exp( cbind(1,ZZ,MM,DD,sqrt(abs(XX))+3 )%*%q2par ) * ( YY - cbind(1,WW,MM,DD,XX)%*%h2par )
                            + cbind(1,WW,XX) %*% h0par )
  
  dataframe <- data.frame( psiboots_POR=psiboots_POR[j], psiboots_PIPW=psiboots_PIPW[j],
                           psiboots_Ph1=psiboots_Ph1[j], psiboots_Ph2=psiboots_Ph2[j],
                           psiboots_PQR=psiboots_PQR[j] )
  return(dataframe)
}

pb <- txtProgressBar(style = 3, char = "#")
start_time <- Sys.time()
library(foreach)
library(doParallel)
num_cores <- detectCores()
cl <- makeCluster(num_cores)
registerDoParallel(cl)

for (i in 1:nsimu) {
  # generate X, U, A, Z, W, M, D, Y
  mean <- c(0.25, 0.25, 0)
  sigma <- matrix(c(0.25, 0, 0.05, 0, 0.25, 0.05, 0.05, 0.05, 1), nrow = 3, ncol = 3)
  XU <- mvrnorm(n, mean, sigma)
  X <- XU[, c(1,2)]
  U <- XU[, 3]
  A <- rbinom(n, 1, 1/(1 + exp(-X%*%c(0.5,0.5) - 0.4*U)) )
  Z <- rnorm(n, 0.2-0.52*A+X%*%c(0.2,0.2)-U, 1)
  W <- rnorm(n, 0.3+X%*%c(0.2,0.2)-0.6*U, 1)
  D <- rnorm(n, -0.3*A-X%*%c(0.5,0.5)+0.4*U, 1)
  M <- rnorm(n, 0.1*A+0.4*D-X%*%c(0.5,0.5)-0.1*U, 2)
  Y <- 2 + 2*A + M + D + 2*W - X%*%c(1,1) - U + 2*rnorm(n, 0, 1)
  data <- list(Z=Z, W=W, A=A, M=M, D=D, X=X, Y=Y) # observed data: Z, W, A, M, D, X, Y
  
  ##### proximal outcome regression ####
  inioptim <- c(0,0,0,0,0,0)
  h2par <- optim(par=inioptim, 
                 fn=bridgeh2, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  inioptim <- c(0,0,0,0,0)
  h1par <- optim(par=inioptim,
                 fn=bridgeh1, parah2=h2par, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  inioptim <- c(0,0,0,0)
  h0par <- optim(par=inioptim,
                 fn=bridgeh0, parah1=h1par, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  
  psi_POR[i] <- mean( cbind(1,W,X) %*% h0par )
  
  #### proximal IPW ####
  inioptim <- c(0,0,0,0)
  q0par <- optim(par=inioptim,
                 fn=bridgeq0_mis, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  inioptim <- c(0,0,0,0,0)
  q1par <- optim(par=inioptim,
                 fn=bridgeq1_mis, paraq0=q0par, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  inioptim <- c(0,0,0,0,0,0)
  q2par <- optim(par=inioptim,
                 fn=bridgeq2_mis, paraq0=q0par, paraq1=q1par, data1=data,
                 method = "BFGS", hessian=FALSE)$par
  
  psi_PIPW[i] <- mean( A*Y * ( 1+ exp(- cbind(1,Z,sqrt(abs(X))+3 )%*%q0par ) ) * exp( cbind(1,Z,D,sqrt(abs(X))+3 )%*%q1par ) * exp( cbind(1,Z,M,D,sqrt(abs(X))+3 )%*%q2par ) )
  
  #### proximal hybrid 1 ####
  psi_Ph1[i] <- mean( A * (cbind(1, W, D, X)%*%h1par) * ( 1+ exp(- cbind(1,Z,sqrt(abs(X))+3 )%*%q0par ) ) )
  
  #### proximal hybrid 2 ####
  psi_Ph2[i] <- mean( (1-A) * (cbind(1, W, M, D, X)%*%h2par) * ( 1+ exp(- cbind(1,Z,sqrt(abs(X))+3 )%*%q0par ) ) * exp( cbind(1,Z,D,sqrt(abs(X))+3 )%*%q1par ) )
  
  #### proximal quadruply robust ####
  psi_PQR[i] <- mean( A * ( 1+ exp(- cbind(1,Z,sqrt(abs(X))+3 )%*%q0par ) ) * ( cbind(1, W, D, X)%*%h1par - cbind(1,W,X) %*% h0par ) 
                      + (1-A) * ( 1+ exp(- cbind(1,Z,sqrt(abs(X))+3 )%*%q0par ) ) * exp( cbind(1,Z,D,sqrt(abs(X))+3 )%*%q1par ) * ( cbind(1, W, M, D, X)%*%h2par - cbind(1, W, D, X)%*%h1par )
                      + A * ( 1+ exp(- cbind(1,Z,sqrt(abs(X))+3 )%*%q0par ) ) * exp( cbind(1,Z,D,sqrt(abs(X))+3 )%*%q1par ) * exp( cbind(1,Z,M,D,sqrt(abs(X))+3 )%*%q2par ) * ( Y - cbind(1, W, M, D, X)%*%h2par )
                      + cbind(1,W,X) %*% h0par )
  
  ###### generate bootstrap samples ######
  boots <- foreach(j=1:nboots) %dopar% fun_boots(j)
  boots <- do.call(rbind, boots)
  
  psiboots_POR <- boots$psiboots_POR
  psiboots_PIPW <- boots$psiboots_PIPW
  psiboots_Ph1 <- boots$psiboots_Ph1
  psiboots_Ph2 <- boots$psiboots_Ph2
  psiboots_PQR <- boots$psiboots_PQR
  
  ##### pivotal confidence intervals ####
  l.POR[i] <- 2*psi_POR[i] - quantile(psiboots_POR, 0.975, na.rm = TRUE)
  u.POR[i] <- 2*psi_POR[i] - quantile(psiboots_POR, 0.025, na.rm = TRUE)
  l.PIPW[i] <- 2*psi_PIPW[i] - quantile(psiboots_PIPW, 0.975, na.rm = TRUE)
  u.PIPW[i] <- 2*psi_PIPW[i] - quantile(psiboots_PIPW, 0.025, na.rm = TRUE)
  l.Ph1[i] <- 2*psi_Ph1[i] - quantile(psiboots_Ph1, 0.975, na.rm = TRUE)
  u.Ph1[i] <- 2*psi_Ph1[i] - quantile(psiboots_Ph1, 0.025, na.rm = TRUE)
  l.Ph2[i] <- 2*psi_Ph2[i] - quantile(psiboots_Ph2, 0.975, na.rm = TRUE)
  u.Ph2[i] <- 2*psi_Ph2[i] - quantile(psiboots_Ph2, 0.025, na.rm = TRUE)
  l.PQR[i] <- 2*psi_PQR[i] - quantile(psiboots_PQR, 0.975, na.rm = TRUE)
  u.PQR[i] <- 2*psi_PQR[i] - quantile(psiboots_PQR, 0.025, na.rm = TRUE)
  
  setTxtProgressBar(pb, i/nsimu)
}

stopCluster(cl)
close(pb)
Sys.time() - start_time

## true value of estimand
psi.true <- 3.28

# Bias
mean(psi_POR) - psi.true
mean(psi_PIPW) - psi.true
mean(psi_Ph1) - psi.true
mean(psi_Ph2) - psi.true
mean(psi_PQR) - psi.true

# MSE
mean( (psi_POR - psi.true)^2 )
mean( (psi_PIPW - psi.true)^2 )
mean( (psi_Ph1 - psi.true)^2 )
mean( (psi_Ph2 - psi.true)^2 )
mean( (psi_PQR - psi.true)^2 )

# Coverage rate
mean( (l.POR<=psi.true) & (u.POR>=psi.true) )
mean( (l.PIPW<=psi.true) & (u.PIPW>=psi.true) )
mean( (l.Ph1<=psi.true) & (u.Ph1>=psi.true) )
mean( (l.Ph2<=psi.true) & (u.Ph2>=psi.true) )
mean( (l.PQR<=psi.true) & (u.PQR>=psi.true) )

# Length
mean( u.POR - l.POR )
mean( u.PIPW - l.PIPW )
mean( u.Ph1 - l.Ph1 )
mean( u.Ph2 - l.Ph2 )
mean( u.PQR - l.PQR )



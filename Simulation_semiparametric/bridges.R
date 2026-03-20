bridgeh2 <- function(para, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  h2 <- cbind(1, W, M, D, X) %*% para
  Q <- cbind(1, Z, M, D, X)
  link <- as.vector( A*(Y - h2) ) * Q
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}

bridgeh1 <- function(para, parah2, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  h1 <- cbind(1, W, D, X) %*% para
  h2 <- cbind(1, W, M, D, X) %*% parah2
  Q <- cbind(1, Z, D, X)
  link <- as.vector( (1-A)*(h2 - h1) ) * Q
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}

bridgeh0 <- function(para, parah1, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  h0 <- cbind(1, W, X) %*% para
  h1 <- cbind(1, W, D, X) %*% parah1
  Q <- cbind(1, Z, X)
  link <- as.vector( A*(h1 - h0) ) * Q
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}


bridgeq0 <- function(para, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  q0 <- 1+ exp(- cbind(1,Z,X) %*% para )
  H <- cbind(1, W, X)
  link <- as.vector( A*q0 - 1 ) * H
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}

bridgeq1 <- function(para, paraq0, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  q0 <- 1+ exp(- cbind(1,Z,X) %*% paraq0 )
  q1 <- q0 * exp( cbind(1,Z,D,X) %*% para )
  H <- cbind(1, W, D, X)
  link <- as.vector( (1-A)*q1 - A*q0 ) * H
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}

bridgeq2 <- function(para, paraq0, paraq1, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  q0 <- 1+ exp(- cbind(1,Z,X) %*% paraq0 )
  q1 <- q0 * exp( cbind(1,Z,D,X) %*% paraq1 )
  q2 <- q1 * exp( cbind(1,Z,M,D,X) %*% para )
  H <- cbind(1, W, M, D, X)
  link <- as.vector( A*q2 - (1-A)*q1 ) * H
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}


###### misspecified bridges #####

#### Experiments 2 and 3 ####
bridgeq0_mis <- function(para, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  q0 <- 1+ exp(- cbind(1,Z,sqrt(abs(X))+3 ) %*% para )
  H <- cbind(1, W, X)
  link <- as.vector( A*q0 - 1 ) * H
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}

bridgeq1_mis <- function(para, paraq0, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  q0 <- 1+ exp(- cbind(1,Z,sqrt(abs(X))+3 ) %*% paraq0 )
  q1 <- q0 * exp( cbind(1,Z,D,sqrt(abs(X))+3 ) %*% para )
  H <- cbind(1, W, D, X)
  link <- as.vector( (1-A)*q1 - A*q0 ) * H
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}

bridgeq2_mis <- function(para, paraq0, paraq1, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  q0 <- 1+ exp(- cbind(1,Z,sqrt(abs(X))+3 ) %*% paraq0 )
  q1 <- q0 * exp( cbind(1,Z,D,sqrt(abs(X))+3 ) %*% paraq1 )
  q2 <- q1 * exp( cbind(1,Z,M,D,sqrt(abs(X))+3 ) %*% para )
  H <- cbind(1, W, M, D, X)
  link <- as.vector( A*q2 - (1-A)*q1 ) * H
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}


bridgeh2_mis <- function(para, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  h2 <- cbind(1, W, M, D, sqrt(abs(X))+3 ) %*% para
  Q <- cbind(1, Z, M, D, X)
  link <- as.vector( A*(Y - h2) ) * Q
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}

bridgeh1_mis <- function(para, parah2, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  h1 <- cbind(1, W, D, sqrt(abs(X))+3 ) %*% para
  h2 <- cbind(1, W, M, D, sqrt(abs(X))+3 ) %*% parah2
  Q <- cbind(1, Z, D, X)
  link <- as.vector( (1-A)*(h2 - h1) ) * Q
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}

bridgeh0_mis <- function(para, parah1, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  h0 <- cbind(1, W, sqrt(abs(X))+3 ) %*% para
  h1 <- cbind(1, W, D, sqrt(abs(X))+3 ) %*% parah1
  Q <- cbind(1, Z, X)
  link <- as.vector( A*(h1 - h0) ) * Q
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}

#### Experiment 4 ####
bridgeh0_e4 <- function(para, parah1, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  h0 <- cbind(1, W, sqrt(abs(X))+3 ) %*% para
  h1 <- cbind(1, W, D, X ) %*% parah1
  Q <- cbind(1, Z, X)
  link <- as.vector( A*(h1 - h0) ) * Q
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}


bridgeq1_e4 <- function(para, paraq0, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  q0 <- 1+ exp(- cbind(1,Z,X ) %*% paraq0 )
  q1 <- q0 * exp( cbind(1,Z,D,sqrt(abs(X))+3 ) %*% para )
  H <- cbind(1, W, D, X)
  link <- as.vector( (1-A)*q1 - A*q0 ) * H
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}

bridgeq2_e4 <- function(para, paraq0, paraq1, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  q0 <- 1+ exp(- cbind(1,Z,X ) %*% paraq0 )
  q1 <- q0 * exp( cbind(1,Z,D,sqrt(abs(X))+3 ) %*% paraq1 )
  q2 <- q1 * exp( cbind(1,Z,M,D,sqrt(abs(X))+3 ) %*% para )
  H <- cbind(1, W, M, D, X)
  link <- as.vector( A*q2 - (1-A)*q1 ) * H
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}


#### Experiment 5 ####
bridgeh1_e5 <- function(para, parah2, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  h1 <- cbind(1, W, D, sqrt(abs(X))+3 ) %*% para
  h2 <- cbind(1, W, M, D, X ) %*% parah2
  Q <- cbind(1, Z, D, X)
  link <- as.vector( (1-A)*(h2 - h1) ) * Q
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}

bridgeh0_e5 <- function(para, parah1, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  h0 <- cbind(1, W, sqrt(abs(X))+3 ) %*% para
  h1 <- cbind(1, W, D, sqrt(abs(X))+3 ) %*% parah1
  Q <- cbind(1, Z, X)
  link <- as.vector( A*(h1 - h0) ) * Q
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}


bridgeq2_e5 <- function(para, paraq0, paraq1, data1){
  A <- as.matrix(data1$A); Y <- as.matrix(data1$Y)
  Z <- as.matrix(data1$Z); W <- as.matrix(data1$W)
  X <- as.matrix(data1$X); M <- as.matrix(data1$M); D <- as.matrix(data1$D)
  
  q0 <- 1+ exp(- cbind(1,Z,X ) %*% paraq0 )
  q1 <- q0 * exp( cbind(1,Z,D,X ) %*% paraq1 )
  q2 <- q1 * exp( cbind(1,Z,M,D,sqrt(abs(X))+3 ) %*% para )
  H <- cbind(1, W, M, D, X)
  link <- as.vector( A*q2 - (1-A)*q1 ) * H
  link0 <- colMeans(link, na.rm = TRUE)
  loss <- sum(link0 ^2)
  return(loss)
}


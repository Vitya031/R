# install.packages("neuralnet")
# install.packages("boot")


Sys.setenv(TZ = 'GMT')


set.seed(500)
data <- read.csv("C:/Users/vitek_000/Documents/Semester_9_lessons/applied_statistics/Labs2875339/insurance_4lab.csv")


apply(data,2,function(x) sum(is.na(x)))

# crim      zn   indus    chas     nox      rm     age     dis     rad     tax ptratio 
# 0       0       0       0       0       0       0       0       0       0       0 
# black   lstat    medv 
# 0       0       0 


data$smoker[data$smoker=='no']<-0
data$smoker[data$smoker=='yes']<-1
data$region[data$region=='northeast']<-10
data$region[data$region=='northwest']<-20
data$region[data$region=='southeast']<-30
data$region[data$region=='southwest']<-40
data$sex[data$sex=='female']<-0
data$sex[data$sex=='male']<-1

data$smoker<-as.numeric(data$smoker)
data$region<-as.numeric(data$region)
data$sex<-as.numeric(data$sex)


index <- sample(1:nrow(data),round(0.75*nrow(data)))
maxs.lm <- apply(data, 2, max) 
mins.lm <- apply(data, 2, min)

scaled.lm <- as.data.frame(scale(data, center = mins.lm, scale = maxs.lm - mins.lm))

train_lm <- scaled.lm[index,]
test_lm <- scaled.lm[-index,]
# train <- data[index,]
# test <- data[-index,]
lm.fit <- glm(charges~., data=train_lm)
summary(lm.fit)
pr.lm <- predict(lm.fit,test_lm)
MSE.lm <- sum((pr.lm - test_lm$charges)^2)/nrow(test_lm)


maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)

scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

train_ <- scaled[index,]
test_ <- scaled[-index,]


library(neuralnet)
n <- names(train_)
f <- as.formula(paste("charges ~", paste(n[!n %in% "charges"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=4,linear.output=T)


plot(nn)


pr.nn <- compute(nn,test_[,1:6])

pr.nn_ <- pr.nn$net.result*(max(data$charges)-min(data$charges))+min(data$charges)
test.r <- (test_$charges)*(max(data$charges)-min(data$charges))+min(data$charges)

MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)


print(paste(MSE.lm,MSE.nn))

# "21.6297593507225 10.1542277747038"


par(mfrow=c(1,2))

plot(test_lm$charges,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

plot(test_lm$charges,pr.lm,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)


plot(test_lm$charges,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
points(test_lm$charges,pr.lm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,col=c('red','blue'))


library(boot)
set.seed(200)
lm.fit <- glm(charges~.,data=data)
cv.glm(data,lm.fit,K=10)$delta[1]

# 23.83560156


set.seed(450)
cv.error <- NULL
k <- 10

library(plyr) 
pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  index <- sample(1:nrow(data),round(0.9*nrow(data)))
  train.cv <- scaled[index,]
  test.cv <- scaled[-index,]
  
  nn <- neuralnet(f,data=train.cv,hidden=c(4),linear.output=T)
  
  pr.nn <- compute(nn,test.cv[,1:6])
  pr.nn <- pr.nn$net.result*(max(data$charges)-min(data$charges))+min(data$charges)
  
  test.cv.r <- (test.cv$charges)*(max(data$charges)-min(data$charges))+min(data$charges)
  
  cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
  
  pbar$step()
}


mean(cv.error)
# 10.32697995

cv.error
# 17.640652805  6.310575067 15.769518577  5.730130820 10.520947119  6.121160840
# 6.389967211  8.004786424 17.369282494  9.412778105


boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)



# Проба:
# library(psych)
# pairs.panels(train_lm[,-7], gap = 0, pch=21)


install.packages("installr")
library(installr)
updateR()
R.Version()

x <- bar51$x
plot(density(x))
mem <- kmeans(x,2)$cluster
mu1 <- mean(x[mem==1])
mu2 <- mean(x[mem==2])
sigma1 <- sd(x[mem==1])
sigma2 <- sd(x[mem==2])
pi1 <- sum(mem==1)/length(mem)
pi2 <- sum(mem==2)/length(mem)
# modified sum only considers finite values
sum.finite <- function(x) {
  sum(x[is.finite(x)])
}

Q <- 0
# starting value of expected value of the log likelihood
Q[2] <- sum.finite(log(pi1)+log(dnorm(x, mu1, sigma1))) + sum.finite(log(pi2)+log(dnorm(x, mu2, sigma2)))

k <- 2

while (abs(Q[k]-Q[k-1])>=1e-6) {
  # E step
  comp1 <- pi1 * dnorm(x, mu1, sigma1)
  comp2 <- pi2 * dnorm(x, mu2, sigma2)
  comp.sum <- comp1 + comp2
  
  p1 <- comp1/comp.sum
  p2 <- comp2/comp.sum
  
  # M step
  pi1 <- sum.finite(p1) / length(x)
  pi2 <- sum.finite(p2) / length(x)
  
  mu1 <- sum.finite(p1 * x) / sum.finite(p1)
  mu2 <- sum.finite(p2 * x) / sum.finite(p2)
  
  sigma1 <- sqrt(sum.finite(p1 * (x-mu1)^2) / sum.finite(p1))
  sigma2 <- sqrt(sum.finite(p2 * (x-mu2)^2) / sum.finite(p2))
  
  p1 <- pi1 
  p2 <- pi2
  
  k <- k + 1
  Q[k] <- sum(log(comp.sum))
}

library(mixtools)
gm<-normalmixEM(x,k=2,lambda=c(0.9,0.1),mu=c(0.4,0.3),sigma=c(0.05,0.02))
## number of iterations= 84
gm$mu
## [1] 0.4463909 0.3312405
gm$sigma
## [1] 0.05483107 0.01948010
gm$lambda  # posterior probabilities
## [1] 0.8390532 0.1609468

hist(x, prob=T, breaks=32, xlim=c(range(x)[1], range(x)[2]), main='')
lines(density(x), col="green", lwd=2)
x1 <- seq(from=range(x)[1], to=range(x)[2], length.out=1000)
y <- pi1 * dnorm(x1, mean=mu1, sd=sigma1) + pi2 * dnorm(x1, mean=mu2, sd=sigma2)
lines(x1, y, col="red", lwd=2)
legend('topright', col=c("green", 'red'), lwd=2, legend=c("kernal", "fitted"))


library(mixtools)
gm <- normalmixEM(x,k=2,lambda=c(0.5,0.5),mu=c(0.3,0.4),sigma=c(0.05,0.06))
## number of iterations= 154
gm$mu
## [1] 0.2521802 0.4058755
gm$sigma
## [1] 0.04744476 0.07862689
gm$lambda  # posterior probabilities
## [1] 0.4914139 0.5085861


####  WORKING ####  

library(readxl)
Auto_Insurance_Claims_Sample_1 <- read_excel("Auto_Insurance_Claims_Sample 1.xlsx")
data=Auto_Insurance_Claims_Sample_1
plot(density(data$`Claim Amount`))
plot(density(data$`Total Claim Amount`))

## 3 DISTRIBUTION I.E X1,X2,X3 ~ LOGNORMAL DISTRIBUTION ##

?kmeans
mem <- kmeans(data$`Total Claim Amount`,3)$cluster
head(mem)
mu1 <- mean(data$`Total Claim Amount`[mem==1])
mu1
mu2 <- mean(data$`Total Claim Amount`[mem==2])
mu2
mu3 <- mean(data$`Total Claim Amount`[mem==3])
mu3
sigma1 <- sd(data$`Total Claim Amount`[mem==1])
sigma1
sigma2 <- sd(data$`Total Claim Amount`[mem==2])
sigma2
sigma3 <- sd(data$`Total Claim Amount`[mem==3])
sigma3
tau1 <- sum(mem==1)/length(mem)
tau1
tau2 <- sum(mem==2)/length(mem)
tau2
tau3 <- sum(mem==3)/length(mem)
tau3

mu_old=c(mu1,mu2,mu3)
tau_old=c(tau1,tau2,tau3)
sigma_old=c(sigma1,sigma2,sigma3)
initial_parameters=data.frame(mu_old,sigma_old,tau_old)

x=data$`Total Claim Amount`
sum.finite <- function(x) {
  sum(x[is.finite(x)])
}

Q[2] <- 0
# starting value of expected value of the log likelihood
Q[3] <- sum.finite(log(tau1)+log(dlnorm(x, mu1, sigma1))) +
  sum.finite(log(tau2)+log(dlnorm(x, mu2, sigma2))) +
  sum.finite(log(tau3)+log(dlnorm(x, mu3, sigma3)))

k <- 3

while (abs(Q[k]-Q[k-1])>=0.000001) {
  # E step
  comp1 <- tau1 * dlnorm(x, mu1, sigma1)
  comp2 <- tau2 * dlnorm(x, mu2, sigma2)
  comp3 <- tau3 * dlnorm(x, mu3, sigma3)
  comp.sum <- comp1 + comp2+comp3
  
  p1 <- comp1/comp.sum
  p2 <- comp2/comp.sum
  p3 =  comp3/comp.sum
  # M step
  tau1 <- sum.finite(p1) / length(x)
  tau2 <- sum.finite(p2) / length(x)
  tau3  <- sum.finite(p3) / length(x)
  mu1 <- sum.finite(p1 * log(x)) / sum.finite(p1)
  mu2 <- sum.finite(p2 * log(x)) / sum.finite(p2)
  mu3 <- sum.finite(p3 * log(x)) / sum.finite(p3)
  sigma1 <- sqrt(sum.finite(p1 * (x-mu1)^2) / sum.finite(p1))
  sigma2 <- sqrt(sum.finite(p2 * (x-mu2)^2) / sum.finite(p2))
  sigma3 <- sqrt(sum.finite(p3 * (x-mu3)^2) / sum.finite(p3))
  
  p1 <- tau1 
  p2 <- tau2
  p3 = tau3
  k <- k + 1
  Q[k] <- sum(log(comp.sum))
  }

mu_new=c(mu1,mu2,mu3)
tau_new=c(tau1,tau2,tau3)
sigma_new=c(sigma1,sigma2,sigma3)
final_parameters=data.frame(mu_new,sigma_new,tau_new)

#### WORKING 2  ####
library(readxl)
Auto_Insurance_Claims_Sample_1 <- read_excel("Auto_Insurance_Claims_Sample 1.xlsx")
data1 = Auto_Insurance_Claims_Sample_1
x = data1$`Total Claim Amount`
colSums(is.na(data1))
sum(is.na(data1))

# no null values
?plot
plot(density(x),col="blue",main ="" ,lwd=3,xlab = "Claim Amount",ylab = "Density")
hist(x,breaks=40)
# It seems like there are 3 components of log normal distributions
# For initial values use K means clustering
mem = kmeans(x,3)$cluster
head(mem)
data1$Cluster= mem
View(data1)


mu1 = mean(x[mem==1])
mu2 = mean(x[mem==2])
mu3 = mean(x[mem ==3])

sigma1 = sd(x[mem==1])
sigma2 = sd(x[mem==2])
sigma3 = sd(x[mem==3])

tau1 = sum(mem==1)/length(mem)
tau2 = sum(mem==2)/length(mem)
tau3 = sum(mem==3)/length(mem)

mu1
mu2
mu3

sigma1
sigma2
sigma3

tau1
tau2
tau3

mu = c(mu1,mu2,mu3)
sigma = c(sigma1, sigma2, sigma3)
tau = c(tau1, tau2, tau3)

initial_values = data.frame(mu,sigma,tau)
initial_values
# modified sum only considers finite values

sum.finite <- function(x) {
  
  sum(x[is.finite(x)])
  
}

Q=0
Q[2] = 0
# starting value of expected value of the log likelihood
Q[3] = sum.finite(log(tau1)+log(dlnorm(x, mu1, sigma1))) + sum.finite(log(tau2)+log(dlnorm(x, mu2, sigma2))) + sum.finite(log(tau3)+log(dlnorm(x, mu3, sigma3)))
k = 3
while (abs(Q[k]-Q[k-1])>=1e-2) {
  
  # E step
  
  comp1 = tau1 * dlnorm(x, mu1, sigma1)
  
  comp2 = tau2 * dlnorm(x, mu2, sigma2)
  
  comp3 = tau3 * dlnorm(x, mu3, sigma3)
  
  comp.sum = comp1 + comp2 + comp3
  
  
  T1 = comp1/comp.sum
  
  T2 = comp2/comp.sum
  
  T3 = comp3/comp.sum
  
  
  # M step
  
  tau1 = sum.finite(T1) / length(x)
  
  tau2 = sum.finite(T2) / length(x)
  
  tau3 = sum.finite(T3) / length(x)
  
  
  mu1 <- sum.finite(T1 * log(x)) / sum.finite(T1)
  
  mu2 <- sum.finite(T2 * log(x)) / sum.finite(T2)
  
  mu3 = sum.finite(T3 * log(x)) / sum.finite(T3)
  
  
  sigma1 <- sqrt(sum.finite(T1 * (log(x)-mu1)^2) / sum.finite(T1))
  
  sigma2 <- sqrt(sum.finite(T2 * (log(x)-mu2)^2) / sum.finite(T2))
  
  sigma3 = sqrt(sum.finite(T3 * (log(x)-mu3)^2) / sum.finite(T3))
  
  
  T1 <- tau1
  
  T2 <- tau2
  
  T3 = tau3
  
  
  k <- k + 1
  
  Q[k] <- sum(log(comp.sum))
  
}

mu1
mu2
mu3
exp(mu1)
exp(mu2)
exp(mu3)

plot(density(x),col="blue",main ="" ,lwd=2,xlab = "Claim Amount",ylab = "Density")
abline(v=exp(mu1),col="green")
abline(v=exp(mu2),col="yellow")
abline(v=exp(mu3),col="red")

exp(sigma1)
exp(sigma2)
exp(sigma3)

tau1
tau2
tau3
##  exp(tau1)+exp(tau2)+exp(tau3)  ## 4.2879
tau=tau1+tau2+tau3
tau

mu_new = c(exp(mu1),exp(mu2),exp(mu3))
sigma_new =c(exp(sigma1),exp(sigma2),exp(sigma3))
tau_new = c(tau1,tau2,tau3)

final_values = data.frame(mu_new,sigma_new,tau_new)
final_values
initial_values

data1$Customer=as.factor(data1$Customer)
data1$Country=as.factor(data1$Country)
data1$`State Code`=as.factor(data1$`State Code`)
data1$State=as.factor(data1$State)
data1$`Effective To Date`=as.factor(data1$`Effective To Date`)
data1$`Claim Amount`=as.factor(data1$`Claim Amount`)
data1$Income=as.factor(data1$Income)
data1$`Location Code`=as.factor(data1$`Location Code`)
data1$`Marital Status`=as.factor(data1$`Marital Status`)
data1$`Monthly Premium Auto`=as.factor(data1$`Marital Status`)
data1$`Months Since Last Claim`=as.factor(data1$`Months Since Last Claim`)
data1$`Months Since Policy Inception`=as.factor(data1$`Months Since Policy Inception`)
data1$`Number of Open Complaints`=as.factor(data1$`Number of Open Complaints`)
data1$`Number of Policies`=as.factor(data1$`Number of Policies`)
data1$Policy=as.factor(data1$Policy)
data1$`Sales Channel`=as.factor(data1$`Sales Channel`)
data1$Education = as.factor(data1$Education)
data1$Response = as.factor(data1$Response)
data1$EmploymentStatus = as.factor(data1$EmploymentStatus)
data1$Coverage = as.factor(data1$Coverage)
data1$Gender = as.factor(data1$Gender)
data1$`Vehicle Size`= as.factor(data1$`Vehicle Size`)
data1$`Vehicle Class` = as.factor(data1$`Vehicle Class`)
data1$`Claim Reason` = as.factor(data1$`Claim Reason`)
data1$`Policy Type` = as.factor(data1$`Policy Type`)

str(data1)
attach(data1)
anova(lm(data1$`Total Claim Amount` ~ data1$Customer + data1$EmploymentStatus
         
         + data1$Country +data1$`State Code`+ data1$State + data1$`Claim Amount`
         
         + data1$Response +data1$Coverage +data1$Education +data1$`Effective To Date`
         
         +data1$EmploymentStatus+data1$Gender+data1$Income+data1$`Location Code`
        
         data1$`Claim Reason` + data1$`Policy Type`+data1$`Marital Status`+
           
         data1$`Monthly Premium Auto`+data1$`Months Since Last Claim`
         
         +data1$`Months Since Policy Inception`+data1$`Number of Open Complaints`

         +data1$`Number of Policies`+data1$Policy+data1$`Sales Channel`+
  
         data1$`Vehicle Class`+data1$`Vehicle Size`))

data1$Income=as.numeric(data1$Income)
cor(data1$`Total Claim Amount`,data1$Income)
plot(data1$Income,data1$`Total Claim Amount`)



#### INSURANCE CLAIM CSV####
data1=insurance_claims
x = data1$total_claim_amount
sum(is.na(data1))

# no null values
plot(density(x),col="blue",main ="" ,lwd=3,xlab = "Claim Amount",ylab = "Density")
hist(x,breaks=40)
# It seems like there are 3 components of log normal distributions
# For initial values use K means clustering
mem = kmeans(x,2)$cluster
head(mem)
data1$Cluster= mem

mu1 = mean(x[mem==1])
mu2 = mean(x[mem==2])
#mu3 = mean(x[mem ==3])

sigma1 = sd(x[mem==1])
sigma2 = sd(x[mem==2])
#sigma3 = sd(x[mem==3])

tau1 = sum(mem==1)/length(mem)
tau2 = sum(mem==2)/length(mem)
#tau3 = sum(mem==3)/length(mem)

mu = c(mu1,mu2)
sigma = c(sigma1, sigma2)
tau = c(tau1, tau2)

initial_values = data.frame(mu,sigma,tau)
initial_values
# modified sum only considers finite values

sum.finite <- function(x) {
  
  sum(x[is.finite(x)])
  
}

Q=0
#Q[2] = 0
# starting value of expected value of the log likelihood
Q[2] = sum.finite(log(tau1)+log(dlnorm(x, mu1, sigma1))) + sum.finite(log(tau2)+log(dlnorm(x, mu2, sigma2))) 
k = 2
while (abs(Q[k]-Q[k-1])>=1e-5) {
  
  # E step
  
  comp1 = tau1 * dlnorm(x, mu1, sigma1)
  comp2 = tau2 * dlnorm(x, mu2, sigma2)
  
  comp.sum = comp1 + comp2 

  T1 = comp1/comp.sum
  T2 = comp2/comp.sum
  
  # M step
  tau1 = sum.finite(T1) / length(x)
  tau2 = sum.finite(T2) / length(x)
  
  mu1 <- sum.finite(T1 * log(x)) / sum.finite(T1)
  mu2 <- sum.finite(T2 * log(x)) / sum.finite(T2)
  
  sigma1 <- sqrt(sum.finite(T1 * (log(x)-mu1)^2) / sum.finite(T1))
  sigma2 <- sqrt(sum.finite(T2 * (log(x)-mu2)^2) / sum.finite(T2))
  
  T1 <- tau1
  T2 <- tau2
  
  k <- k + 1
  Q[k] <- sum(log(comp.sum))
}

mu1
mu2

exp(mu1)
exp(mu2)

plot(density(x),col="blue",main ="" ,lwd=2,xlab = "Claim Amount",ylab = "Density")
abline(v=exp(mu1),col="green")
abline(v=exp(mu2),col="yellow")

exp(sigma1)
exp(sigma2)

tau1
tau2

##  exp(tau1)+exp(tau2)+exp(tau3)  ## 4.2879
tau=tau1+tau2
tau

mu_new = c(exp(mu1),exp(mu2))
sigma_new =c(exp(sigma1),exp(sigma2))
tau_new = c(tau1,tau2)

final_values = data.frame(mu_new,sigma_new,tau_new)
final_values
initial_values

### NOMRAL MIX
u=normalmixEM(x,tau,mu,sigma,k=2)
summary(u)


u$lambda
u$mu
u$sigma

plot(density(x))
abline(v=u$mu[1],col="green")
abline(v=u$mu[2],col="red")

v=rnormmix(9134,u$lambda,u$mu,u$sigma)
plot(ecdf(x))
lines(ecdf(v),col="red")

ks.test(x,v)


#### BOOTSTRAP####

x=sample(data1$`Total Claim Amount`,9134,replace = T)

x = data1$`Total Claim Amount`
y=data1$Cluster
mean.x=mean(x[which(y==1)])
mean.x
est = rep(0,1000)

set.seed(111)
for(i in 1:1000){
  x = rlnorm(4,mean=mean.x,sd=10)
  est[i]=mean(x)
}

mean.est = mean(est)
mean.est
sd.est = sd(est)
sd.est






library(boot)
b <- function(x,i)
{ 
  mean = mean(x[i])
  #s.d = sd(x[i])
  return(c(mean))
}
?boot
bs <- boot(x, b, R = 1000)
bs


n=9134
u=sum(log(x)/n)
mle1=c(mu1,sigma1)
mle1
est=rep(0,5000)
diff=rep(0,5000)
set.seed(111)
for(i in 1:5000){
  x=rexp(10,rate=mle)
  est[i]=1/mean(x)
  diff[i]=abs(est[i]-mle)
}
mean_1=mean(est)
sd_1=sd(est)
# mean(diff)
# head(diff)

n=9134
mle2=c(mu2,sigma2)
mle2
est=rep(0,5000)
diff=rep(0,5000)
set.seed(111)
for(i in 1:5000){
  x=rexp(10,rate=mle2)
  est[i]=1/mean(x)
  diff[i]=abs(est[i]-mle2)
}
mean_2=mean(est)
sd_2=sd(est)
#mean(diff)
#head(diff)

mle3=c(mu3,sigma3)
mle3
est=rep(0,5000)
diff=rep(0,5000)
set.seed(111)
for(i in 1:5000){
  x=rexp(10,rate=mle3)
  est[i]=1/mean(x)
  diff[i]=abs(est[i]-mle3)
}
mean_3=mean(est)
sd_3=sd(est)

data.frame(mean_1,mean_2,mean_3,mu1,mu2,mu3)
#The function boot sends data and random number vector to function f
#1000 times.Each time random number vector is different.

install.packages("mixR")
library(mixR)
y =rmixlnorm(1000,tau_new,mu_new,sigma_new)

plot(y,"l",col = "red")
hist(y,breaks = 40)

#### GRAPHS ####
install.packages("ggplot2") 
install.packages("data.table") 
library(data.table) 
library(ggplot2) 
library(dplyr) 

View(data1)
##data=data.table(Auto_Insurance_Claims_Sample_1,data1$Cluster)
data =data.table(Auto_Insurance_Claims_Sample_1) 
class(data) 
cat("Data has unique Rows :",uniqueN(data)==nrow(data))  
#Data has unique rows 

# Frequency per state 
ggplot(data,aes(State))+ geom_bar(fill=rainbow(5))#+geom_col(fill="coral")
#x axis : States , y axis: No. of customers from that paticular state 
#Interpretation: Most of the customers who have claimed are from Missouri state 


# Policy type Freq 
data%>%ggplot(aes(`Policy Type`))+geom_bar(fill=rainbow(3)) 
#Interpretation :  Most of the customers who have claimed the policy have personal auto policy type 

# Gender Freq 
data%>%ggplot(aes(Gender))+geom_bar(fill=c("pink", "lightblue")) 
#Interpretation : Number of males & females who have claimed the policy are almost same 

# Claim reason Freq 
data%>%ggplot(aes(`Claim Reason`))+geom_bar(fill=rainbow(4)) 
#Interpretation:  Most of the policies are claimed due to collision  

#Monthly Premium distribution 
data%>%ggplot(aes(`Monthly Premium Auto`))+
  geom_histogram(bins=12,fill='pink',colour="black") 
#Interpretation : Most of the customers pay premium in the range 0 to 50 dollars 

# Claim amount distribution per reason class 
p=ggplot(data,aes(`Total Claim Amount`),binwidth = 50)+
  geom_histogram(fill='light blue',colour="green",stat='bin',bins=10) 
p+facet_grid(cols = vars(`Claim Reason`)) 
#Interpretation: Majority of the total claim amount in 
#all the cases is in the range 250 to 500 dollars 

#Vehicle Class Freq 
data%>%ggplot(aes(`Vehicle Class`))+geom_histogram(bins=12,
                                fill='pink',colour="black",stat="count") 
#Interpretation: Most of the customers who have claimed the policy own a four door car 
#Four door car should have higher premium amount 

#Employment Status Freq 
data%>%ggplot(aes(`EmploymentStatus`))+geom_histogram(fill='pink',colour="black",stat="count") 
#Interpretation : (ask sir) 

#Sales Channel 
data%>%ggplot(aes(`Sales Channel`))+geom_histogram(bins=12,
                         fill='pink',colour="black",stat="count") 
#Interpretation :  Most of the policies that the company gets is through agent 
#If company wants to increase the sale of their policies
#then they should hire more agents 

#ask sir about income & education 
#### BOOTSTRAP R DATA ####
library(MASS)
data(Boston)
View(Boston)
nox<-Boston[,5]
rm<-Boston[,6]
dis<-Boston[,8]
pr<-Boston[,11]
ls<-Boston[,13]
medv<-Boston[,14]
data.boston<- data.frame(medv, nox, rm, dis, pr, ls)
lm.boston<-lm(medv~nox+rm+dis+pr+ls,data=data.boston)
summary(lm.boston)
boot.huber<-function(data,indices,maxit=20){
  data<-data[indices,]
  mod<-lm(medv~nox+rm+dis+pr+ls,data=data, maxit=maxit)
  coefficients(mod)}
library(boot)
boston.boot<-boot(data.boston, boot.huber,1999,maxit=100)
boston.boot

ci.b0<-boot.ci(boston.boot,index=1,type=c("norm","perc","bca"))
ci.nox<-boot.ci(boston.boot,index=2,type=c("norm","perc","bca"))
ci.rm<-boot.ci(boston.boot,index=3,type=c("norm","perc","bca"))
ci.dis<-boot.ci(boston.boot,index=4,type=c("norm","perc","bca"))
ci.pr<-boot.ci(boston.boot,index=5,type=c("norm","perc","bca"))
ci.ls<-boot.ci(boston.boot,index=6,type=c("norm","perc","bca"))
normal.ci<-cbind(c(ci.b0$normal), c(ci.nox$normal),c(ci.rm$normal),
                   c(ci.dis$normal),c(ci.pr$normal),c(ci.ls$normal))
rownames(normal.ci) <- c("Level.Normal","Lower","Upper")
colnames(normal.ci) <- c( "B0","B1","B2","B3","B4","B5")
print(normal.ci)

#### mixture normal  ####

library(readxl)
Auto_Insurance_Claims_Sample_1 <- read_excel("Auto_Insurance_Claims_Sample 1.xlsx")
data1 = Auto_Insurance_Claims_Sample_1
x = data1$`Total Claim Amount`
colSums(is.na(data1))
sum(is.na(data1))

# no null values
?plot
plot(density(x),col="blue",main ="" ,lwd=3,xlab = "Claim Amount",ylab = "Density")
hist(x,breaks=40)
# It seems like there are 3 components of log normal distributions
# For initial values use K means clustering
mem = kmeans(x,3)$cluster
head(mem)
data1$Cluster= mem
View(data1)


mu1 = mean(x[mem==1])
mu2 = mean(x[mem==2])
mu3 = mean(x[mem ==3])

sigma1 = sd(x[mem==1])
sigma2 = sd(x[mem==2])
sigma3 = sd(x[mem==3])

tau1 = sum(mem==1)/length(mem)
tau2 = sum(mem==2)/length(mem)
tau3 = sum(mem==3)/length(mem)

mu = c(mu1,mu2,mu3)
sigma = c(sigma1, sigma2, sigma3)
tau = c(tau1, tau2, tau3)

initial_values = data.frame(mu,sigma,tau)
initial_values

library(mixtools)
u=normalmixEM(x,tau,mu,sigma,k=3)
summary(u)
#plot(u,density = T)


u$lambda
u$mu
u$sigma

plot(density(x))
abline(v=u$mu[1],col="green")
abline(v=u$mu[2],col="red")
abline(v=u$mu[3],col="blue")

# u=normalmixEM(x,tau,mu,sigma)
v=rnormmix(9134,u$lambda,u$mu,u$sigma)
plot(ecdf(x))
lines(ecdf(v),col="red")

# plot(density(x))
# lines(density(v),col="red")


#### mixture gamma ####
mem = kmeans(x,3)$cluster
head(mem)
data1$Cluster= mem
View(data1)

mu1 = mean(x[mem==1])
mu2 = mean(x[mem==2])
mu3 = mean(x[mem ==3])

sigma1 = sd(x[mem==1])
sigma2 = sd(x[mem==2])
sigma3 = sd(x[mem==3])

tau1 = sum(mem==1)/length(mem)
tau2 = sum(mem==2)/length(mem)
tau3 = sum(mem==3)/length(mem)

mu1
mu2
mu3

sigma1
sigma2
sigma3

tau1
tau2
tau3

mu = c(mu1,mu2,mu3)
sigma = c(sigma1, sigma2, sigma3)
tau = c(tau1, tau2, tau3)

initial_values = data.frame(mu,sigma,tau)
initial_values
# modified sum only considers finite values


w=gammamixEM(x, lambda = tau, alpha = c(4.1528,13.986,22.91801),
           beta = c(0.0178,0.01207,0.0413897), k = 3,
           mom.start = TRUE, fix.alpha = FALSE, epsilon = 1e-02, 
           maxit = 1000, maxrestarts = 20, verb = FALSE)
w$lambda
w$gamma.pars
mean1=w$gamma.pars[1,1]/w$gamma.pars[2,1]
mean2=w$gamma.pars[1,2]/w$gamma.pars[2,2]
mean3=w$gamma.pars[1,3]/w$gamma.pars[2,3]
plot(density(x))
abline(v=mean1,col="blue")
abline(v=mean2,col="red")
abline(v=mean3,col="green")

sigma1=sqrt(w$gamma.pars[1,1]/(w$gamma.pars[2,1])^2)
sigma2=sqrt(w$gamma.pars[1,2]/(w$gamma.pars[2,2])^2)
sigma3=sqrt(w$gamma.pars[1,3]/(w$gamma.pars[2,3])^2)

plot(ecdf(x))
lines(ecdf(rmixgamma(9134,w$lambda,c(mean1,mean2,mean3),
                     c(sigma1,sigma2,sigma3))),col="red")






# What is Motor Insurance Policy
Motor Insurance is an insurance that covers the policyholder in case of financial losses resulting from an accident or other damages sustained by the insured vehicle. 
Its primary objective is to provide protection against physical damage resulting from traffic collisions and against liability that could also arise there-from.

## What does the motor insurance policy cover?
Damages and losses, resulting from natural calamities such as earthquake, floods, fire, lightning, landslide, hurricane, etc.
Damages that result from human intervention, including burglary, theft, riots, strike or any other activity born of malicious intent.
Third-party legal liabilities owing to damages (both bodily injuries and death) caused to third-party as well as financial losses to a third-party property.

## Modelling insurance claims  
Modelling insurance claims can reveal valuable information for insurance risk management. 
From the past insurance policy information, insurance claims predictive models can learn patterns of different insurance claim ratios and can be used to predict risk levels of future insurance policies.
A key challenge for the insurance industry is to charge each customer an appropriate price for the risk they represent. Risk varies widely from customer to customer, and a deep understanding of different risk factors helps predict the likelihood and cost of insurance claims. 
The goal of this project is to understand the behavior of the motor insurance claims.

## Approach 
A model is fitted to the data and the estimated parameters for the model are calculated by the EM algorithm.
We will try to use the bootstrap technique to fit the data and show that the bootstrap sample for observation can be applicable to the estimated parameters.
We try to assess the goodness of fit by test & graphical method.

## Dataset link:

## Histogram
![Histogram](https://user-images.githubusercontent.com/70087327/130547499-5cb361b8-5edd-4a20-a709-56dd1377ccc2.jpg)
By seeing the graph, It is reasonable to say that this distribution is a Mixture distribution and it consist of 2 components.

## What is mixture distribtion
A mixture distribution is the distribution formed from the weighted combination of 2 or more components that can be univariate or bivariate.

It can also be defined by the following formulae :

                                𝑓_𝑥 (𝑥)=𝝉_1 𝑓_1 (𝑥) + 𝝉_2 𝑓_2 (𝑥) +…... + 𝝉_𝑘 𝑓_𝑘 (𝑥) 
     
     0 < 𝝉_𝑗  < 1 for j=1,2,..,k and 𝝉_1 + 𝝉_2 + ….. + 𝝉_𝑘 = 1
     where f1, f2, f3 ……. are the component distributions and 𝝉_𝑗 are the mixing weights.

The mixture of distributions is sometime called compounding, which is extremely important as it can provide a superior fit.

## EM Algortihm 
A commonly used tool for estimating the parameters of a mixture model is the Expectation–Maximization (EM) algorithm, which is an iterative procedure that can serve as a maximum-likelihood estimator. 
The EM iteration alternates between performing an expectation (E) step, which creates a function for the expectation of the log-likelihood evaluated using the current estimate for the parameters, and a maximization (M) step, which computes parameters maximizing the expected log-likelihood found on the E step.
The expected value of the log likelihood is recalculated using the new parameters, and is compared to expected value of log-likelihood from the previous step.
This process is repeated until the converting criteria is met.
![image](https://user-images.githubusercontent.com/70087327/130547681-101e9db9-afde-4711-b25f-199f2712eff5.png)

## E-Step
![image](https://user-images.githubusercontent.com/70087327/130547718-d8c4f67b-bdfc-48ff-a062-ebb1120121a1.png)






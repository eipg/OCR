library(kernlab)
library(e1071)
library(mlr)

data <- read.table("letterdata.txt", header=TRUE)
data$y <- as.factor(data$y)

#************************************* k-fold CV *****************************
#For 5-fold CV, create the indices of the observations going into each set
n <- nrow(data)
ind1 <- sample(c(1:n),round(n/5))
ind2 <- sample(c(1:n)[-ind1],round(n/5))
ind3 <- sample(c(1:n)[-c(ind1, ind2)],round(n/5))
ind4 <- sample(c(1:n)[-c(ind1, ind2, ind3)],round(n/5))
ind5 <- setdiff(c(1:n),c(ind1,ind2,ind3,ind4))
k5_ind <- list(ind1,ind2,ind3,ind4, ind5)

#For 10-fold CV
ind1 <- sample(c(1:n),round(n/10))
ind2 <- sample(c(1:n)[-ind1],round(n/10))
ind3 <- sample(c(1:n)[-c(ind1, ind2)],round(n/10))
ind4 <- sample(c(1:n)[-c(ind1, ind2, ind3)],round(n/10))
ind5 <- sample(c(1:n)[-c(ind1, ind2, ind3, ind4)],round(n/10))
ind6 <- sample(c(1:n)[-c(ind1, ind2, ind3, ind4, ind5)],round(n/10))
ind7 <- sample(c(1:n)[-c(ind1, ind2, ind3, ind4, ind5, ind6)],round(n/10))
ind8 <- sample(c(1:n)[-c(ind1, ind2, ind3, ind4, ind5, ind6, ind7)],round(n/10))
ind9 <- sample(c(1:n)[-c(ind1, ind2, ind3, ind4, ind5, ind6, ind7, ind8)],round(n/10))
ind10 <- setdiff(c(1:n),c(ind1,ind2,ind3,ind4,ind5,ind6,ind7,ind8,ind9))
k10_ind <- list(ind1,ind2,ind3,ind4,ind5,ind6,ind7,ind8,ind9,ind10)

#************************************ SELECT C *******************************
res = tuneParams("classif.ksvm",
                 makeClassifTask(data = train.data, target = "y"),
                 makeResampleDesc("CV", iters = 10L),
                 measures=acc,
                 par.set = makeParamSet(
                   makeDiscreteParam("C", values = c(0.1, 0.5, 1, 2, 5, 10))),
                 control = makeTuneControlGrid())

#********************************* Model Fitting *****************************

#********************************* 1.radial basis ****************************
#Set up a vector of 5 numbers to store the classification rates for each set
corr.class.rate_rb5 <- numeric(5)
for (i in 1:5) {
  #For each run, we use one set for test/validation
  test.data <- data[k5_ind[[i]],]
  #The remaining sets are our training data
  train.ind <- setdiff(c(1:n),k5_ind[[i]])
  train.data <- data[train.ind,]
  #Fit the model on the training data
  letter_classifier_rbf <- ksvm(y ~., data = train.data, kernel="rbfdot", C = 10)
  #Predict using the test data
  letter_predictions_rbf <- predict(letter_classifier_rbf, test.data)
  #Calculate the test data correct classification rate
  corr.class.rate_rb5[i] <- sum(letter_predictions_rbf==test.data$y)/nrow(test.data)
}
#average the 5 correct classification rates to find the overall rate
cv.corr.class.rate_rb5 <- mean(corr.class.rate_rb5)
cv.corr.class.rate_rb5


#Set up a vector of 10 numbers to store the classification rates for each set
corr.class.rate_rb10 <- numeric(10)

for (i in 1:10) {
  #For each run, we use one set for test/validation
  test.data <- data[ind[[i]],]
  #The remaining sets are our training data
  train.ind <- setdiff(c(1:n),k10_ind[[i]])
  train.data <- data[train.ind,]
  #Fit the model on the training data
  letter_classifier_rbf <- ksvm(y ~., data = train.data, kernel="rbfdot", C = 10)
  #Predict using the test data
  letter_predictions_rbf <- predict(letter_classifier_rbf, test.data)
  #Calculate the test data correct classification rate
  corr.class.rate_rb10[i] <- sum(letter_predictions_rbf==test.data$y)/nrow(test.data)
}
#average the 10 correct classification rates to find the overall rate
cv.corr.class.rate_rb10 <- mean(corr.class.rate_rb10)
cv.corr.class.rate_rb10


#********************************* 2.polynomial kernel **********************
#Set up a vector of 5 numbers to store the classification rates for each set
corr.class.rate_poly5 <- numeric(5)
for (i in 1:5) {
  #For each run, we use one set for test/validation
  test.data <- data[k5_ind[[i]],]
  #The remaining sets are our training data
  train.ind <- setdiff(c(1:n),k5_ind[[i]])
  train.data <- data[train.ind,]
  #Fit the model on the training data
  letter_classifier_poly <- ksvm(y ~., data = train.data, kernel="polydot", C = 10)
  #Predict using the test data
  letter_predictions_poly <- predict(letter_classifier_poly, test.data)
  #Calculate the test data correct classification rate
  corr.class.rate_poly5[i] <- sum(letter_predictions_poly==test.data$y)/nrow(test.data)
}
#average the 5 correct classification rates to find the overall rate
cv.corr.class.rate_poly5 <- mean(corr.class.rate_poly5)
cv.corr.class.rate_poly5


#Set up a vector of 10 numbers to store the classification rates for each set
corr.class.rate_poly10 <- numeric(10)

for (i in 1:10) {
  #For each run, we use one set for test/validation
  test.data <- data[k10_ind[[i]],]
  #The remaining sets are our training data
  train.ind <- setdiff(c(1:n),k10_ind[[i]])
  train.data <- data[train.ind,]
  #Fit the model on the training data
  letter_classifier_poly <- ksvm(y ~., data = train.data, kernel="polydot", C = 10)
  #Predict using the test data
  letter_predictions_poly <- predict(letter_classifier_poly, test.data)
  #Calculate the test data correct classification rate
  corr.class.rate_poly10[i] <- sum(letter_predictions_poly==test.data$y)/nrow(test.data)
}
#average the 10 correct classification rates to find the overall rate
cv.corr.class.rate_poly10 <- mean(corr.class.rate_poly10)
cv.corr.class.rate_poly10


#********************************* 3.hyperbolic tangent sigmoid **********************
#Set up a vector of 5 numbers to store the classification rates for each set
corr.class.rate_tanh5 <- numeric(5)
for (i in 1:5) {
  #For each run, we use one set for test/validation
  test.data <- data[k5_ind[[i]],]
  #The remaining sets are our training data
  train.ind <- setdiff(c(1:n),k5_ind[[i]])
  train.data <- data[train.ind,]
  #Fit the model on the training data
  letter_classifier_tanh <- ksvm(y ~., data = train.data, kernel="tanhdot", C = 10)
  #Predict using the test data
  letter_predictions_tanh <- predict(letter_classifier_tanh, test.data)
  #Calculate the test data correct classification rate
  corr.class.rate_tanh5[i] <- sum(letter_predictions_tanh==test.data$y)/nrow(test.data)
}
#average the 5 correct classification rates to find the overall rate
cv.corr.class.rate_tanh5 <- mean(corr.class.rate_tanh5)
cv.corr.class.rate_tanh5


#Set up a vector of 10 numbers to store the classification rates for each set
corr.class.rate_tanh10 <- numeric(10)

for (i in 1:10) {
  #For each run, we use one set for test/validation
  test.data <- data[k10_ind[[i]],]
  #The remaining sets are our training data
  train.ind <- setdiff(c(1:n),k10_ind[[i]])
  train.data <- data[train.ind,]
  #Fit the model on the training data
  letter_classifier_tanh <- ksvm(y ~., data = train.data, kernel="tanhdot", C = 10)
  #Predict using the test data
  letter_predictions_tanh <- predict(letter_classifier_tanh, test.data)
  #Calculate the test data correct classification rate
  corr.class.rate_tanh10[i] <- sum(letter_predictions_tanh==test.data$y)/nrow(test.data)
}
#average the 10 correct classification rates to find the overall rate
cv.corr.class.rate_tanh10 <- mean(corr.class.rate_tanh10)
cv.corr.class.rate_tanh10

#********************************* plot ***********************************
library(ggplot2)
results <- data.frame("kernel" = c("RadialBasis","RadialBasis",
                                   "Polynomial","Polynomial",
                                   "HyperbolicTangent","HyperbolicTangent"),
                      "K" = c("5","10","5","10","5","10"),
                      "corr.class.rate" = c(cv.corr.class.rate_rb5,
                                            cv.corr.class.rate_rb10,
                                            cv.corr.class.rate_poly5,
                                            cv.corr.class.rate_poly10,
                                            cv.corr.class.rate_tanh5,
                                            cv.corr.class.rate_tanh10))

p <- ggplot(data = results, mapping = aes(x = K , y = corr.class.rate, fill = kernel)
            )+geom_bar(stat = "identity", position = "dodge"
            )+scale_fill_manual(values =c("olivedrab3","indianred2","skyblue2"))
p
#************************************* further work ***************************
#*

#********************  random forests ********************
library(randomForest)
#********************* 1.SELECT ntree ********************
rf_res = tuneParams("classif.randomForest", 
                    makeClassifTask(data = train.data, target = "y"), 
                    makeResampleDesc("CV", iters = 10L), 
                    measures=acc, 
                    par.set = makeParamSet(
                      makeDiscreteParam("ntree", 
                                        values = c(500, 1000, 1500, 2000, 2500, 3000))),
                    control = makeTuneControlGrid())
#********************* 2.fit model ********************
RandomForest_modle <- randomForest(y ~ .,
                                  data=letters, 
                                  ntree=500,
                                  mtry=2,
                                  importance=TRUE,
                                  na.action=randomForest::na.roughfix,
                                  replace=FALSE)

#********************* 2.plot ********************


#neural networks,
#decision trees,
#LDA


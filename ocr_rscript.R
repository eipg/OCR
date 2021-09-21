library(kernlab)
library(e1071)
library(caret)
#Split the data 50% training, 25% validation and 25% test
letters <- read.table("letterdata.txt", header=TRUE)
letters$y <- as.factor(letters$y)
training_set <- letters[1:10000, ]
valid_set <- letters[10001:15000,]
test_set <- letters[15001:20000, ]


#Fit the classification SVM for different values of C
#and calculate the validation prediction error
#divided into training set & test set
C.sel<- tune.svm(y~.,data=training_set,type="C-classification",kernel="linear",
                 cost=c(0.1,0.5,1,2,5,10))$best.parameters
final.svm<-svm(y~.,data=training_set,kernel="linear",cost=C.sel,
               type="C-classification",decision.values=T)
fitted.train<-attributes(predict(final.svm,training_set,
                                 decision.values = TRUE))$decision.values
C.sel

#linear model
letter_classifier_linear <- ksvm(y ~., data = training_set, kernel="vanilladot")
letter_classifier_linear

letter_predictions_linear <- predict(letter_classifier_linear, test_set)
table(letter_predictions_linear, test_set$y)
agreement<- letter_predictions_linear==test_set$y
table(agreement)
prop.table(table(agreement))

#rbf kernel
#cs = [0.1, 1, 10, 100, 1000]for c in cs: 
#svc = svm.SVC(kernel='rbf', C=c).fit(X, y)
letter_classifier_rbf <- ksvm(y ~., data = training_set, kernel="rbfdot")
#letter_classifier_rbf <- ksvm(y ~., data = training_set, kernel="rbfdot", C=1)
letter_predictions_rbf <- predict(letter_classifier_rbf, test_set)
agreement<- letter_predictions_rbf==test_set$y
table(agreement)
prop.table(table(agreement))

#polynomial kernel (“polydot”)
letter_classifier_poly <- ksvm(y ~., data = training_set, kernel="polydot")
letter_predictions_poly <- predict(letter_classifier_poly, test_set)
agreement<- letter_predictions_poly==test_set$y
table(agreement)
prop.table(table(agreement))

#hyperbolic tangent sigmoid (“tanhdot”). 
letter_classifier_tanh <- ksvm(y ~., data = training_set, kernel="tanhdot")
letter_predictions_tanh <- predict(letter_classifier_tanh, test_set)
agreement<- letter_predictions_tanh==test_set$y
table(agreement)
prop.table(table(agreement))

#plot

#Plot the ROC for the validation data
library(ROCR)
pred <- prediction(fitted.valid,valid.data$sp,label.ordering=c("O","B"))
#label.ordering: change the default ordering of the classes
#by supplying a vector containing the negative and the positive class label
perf <- performance(pred, "tpr", "fpr")
plot(perf, main="Validation data ROC", cex.main=0.75, cex.lab=0.75)

#Try different alternative classification methods, like decision trees, random forests,
#neural networks and Gaussian processes, and compare their test set performance with that 
#obtained with SVNs.
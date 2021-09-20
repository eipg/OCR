library(kernlab)
#divided into training set & test set
letters <- read.table("letterdata.txt", header=TRUE)
letters$y <- as.factor(letters$y)
training_set <- letters[1:16000, ]
test_set <- letters[16001:20000, ]

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

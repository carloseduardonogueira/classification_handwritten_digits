### Classification of handwritten digits ###

install.packages("tidverse")
install.packages("FNN")
install.packages("e1071")
install.packages("rpart")
install.packages("rpart.plot")

library(tidyverse)
library(FNN)
library(rpart)
library(rpart.plot)
library(e1071)


location <- getwd()

#setting working directory
setwd(location)

#loading files
file_list <- list.files(path = location)

#reading the files and creating data frame
df <- data.frame()
class <- c()

for(file in file_list){
  class <- c(class, as.numeric(unlist(strsplit(file, '_'))[1]))
  content_file <- read_lines(file)
  content_file <- content_file[-(1:3)]
  content_file <- as.numeric(unlist(strsplit(content_file, ' ')))
  df <- matrix(unlist(content_file), nrow = nrow(df) + 1, ncol=length(content_file))
  df<-as.data.frame(df)
  df$class <- class
}

#preparing dataframe for classification
idxs <- sample(1:nrow(df), as.integer(0.8*nrow(df)))

train_rows <- df[idxs,]
test_rows <- df[-idxs,]

class_train <- train_rows[,4097]
class_test <- test_rows[,4097]

train <- train_rows[,-4097]
test <- test_rows[,-4097]

#classification using KNN - K-Nearest Neighbors and calculating accuracy 
k <- c(1,3,7,9)
knn_accuracy <- c()

for (k in k) {
  result<-knn(train, test, class_train, k)
  
  hits<-0
  
  for(i in 1:length(class_test)){
    if(result[i] == class_test[i])
      hits <- hits +  1
  }
  knn_accuracy <- c(knn_accuracy, (hits/length(class_test))*100)
}

#classification using decision tree
model <- rpart(class~., train_rows, method = "class", control = rpart.control(minsplit = 1))

plot <- rpart.plot(model, type = 3)

dtree_pred <- predict(model, test, type = "class")

#classification using SVM - Support Vector Machine
classifier = svm(formula = class~.,
                 data = train_rows,
                 type = 'C-classification',
                 kernel = 'linear')

svm_pred = predict(classifier, newdata = test)

#calculating accuracy of SVM and decision tree
svm_hits <- 0
dtree_hits <- 0

for(i in 1:length(class_test)){
  if(svm_pred[i] == class_test[i])
    svm_hits <- svm_hits +  1
  if(dtree_pred[i] == class_test[i])
    dtree_hits <- dtree_hits +  1
}
svm_accuracy <- svm_hits/length(class_test)*100
dtree_accuracy <- dtree_hits/length(class_test)*100

print(knn_accuracy)
print(svm_accuracy)
print(dtree_accuracy)


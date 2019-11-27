############################################
### Classification of handwritten digits ###
############################################
#install.packages("tidyverse")
#install.packages("FNN")
#install.packages("e1071")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("factoextra")
#install.packages("caret")
#install.packages("rgl")

library(rgl)
library(ggplot2)
library(caret)
library(factoextra)
library(tidyverse)
library(FNN)
library(rpart)
library(rpart.plot)
library(e1071)

location <- "/home/carloseduardo/projects/classification_handwritten_digits/data"

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
  print(file)
}

#######################################
#preparing dataframe for classification
#######################################
idxs <- sample(1:nrow(df), as.integer(0.8*nrow(df)))

train_rows <- df[idxs,]
test_rows <- df[-idxs,]

class_train <- train_rows[,4097]
class_test <- factor(test_rows[,4097]) 

train <- train_rows[,-4097]
test <- test_rows[,-4097]

########################################################################
#classification using KNN - K-Nearest Neighbors and calculating accuracy
########################################################################
k1<-knn(train, test, class_train, 1)
confusionMatrix(k1,class_test)

k3<-knn(train, test, class_train, 3)
confusionMatrix(k3,class_test)

k7<-knn(train, test, class_train, 7)
confusionMatrix(k7,class_test)

k9<-knn(train, test, class_train, 9)
confusionMatrix(k9,class_test)

###################################
#classification using decision tree
###################################
model <- rpart(class~., train_rows, method = "class", control = rpart.control(minsplit = 1))

plot <- rpart.plot(model, type = 3)

dtree_pred <- predict(model, test, type = "class")

confusionMatrix(dtree_pred,class_test)

##################################################
#classification using SVM - Support Vector Machine
##################################################
classifier = svm(formula = class~.,
                 data = train_rows,
                 type = 'C-classification',
                 kernel = 'linear')

svm_pred = predict(classifier, newdata = test)

confusionMatrix(svm_pred,class_test)

##############################################
#clustering algorithm
##############################################
df_cluster <- df[,-4097]

kmeans_result <- kmeans(df_cluster, 10)

plot3d(df_cluster, col=kmeans_result$cluster, main="k-means clusters")

fviz_nbclust(df_cluster, kmeans, method = "wss")

results <-table(df$class, kmeans_result$cluster)

clusters <- data_frame()
for(i in 1:10 ){
  cluster <- as.data.frame(t(results[,i]))
  clusters <- rbind(clusters, cluster)
}

ggplot(data = clusters, aes(colnames(clusters),clusters$`0`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 1') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters$`0`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters, aes(colnames(clusters),clusters$`1`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 2') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters$`1`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters, aes(colnames(clusters),clusters$`2`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 3') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters$`2`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters, aes(colnames(clusters),clusters$`3`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 4') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters$`3`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters, aes(colnames(clusters),clusters$`4`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 5') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters$`4`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters, aes(colnames(clusters),clusters$`5`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 6') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters$`5`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters, aes(colnames(clusters),clusters$`6`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 7') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters$`6`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters, aes(colnames(clusters),clusters$`7`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 8') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters$`7`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters, aes(colnames(clusters),clusters$`8`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 9') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters$`8`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters, aes(colnames(clusters),clusters$`9`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 10') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters$`9`), vjust=1.6, color="white", size=3.5)

#############################################
#principal component analisys
#############################################
df.pca <- prcomp(df[,1:4096], center = TRUE,scale. = TRUE)
summary(df.pca)

fviz_eig(df.pca)

new_df <- as.data.frame(predict(df.pca, df)) 
new_df$class <- df$class

##############################################
#dataset split after PCA application
##############################################
idxs_pca <- sample(1:nrow(new_df), as.integer(0.8*nrow(new_df)))

train_rows_pca <- new_df[idxs_pca,]
test_rows_pca <- new_df[-idxs_pca,]

class_train_pca <- train_rows_pca[,1950]
class_test_pca <- factor(test_rows_pca[,1950]) 

train_pca <- train_rows_pca[,-1950]
test_pca <- test_rows_pca[,-1950]

##############################################
#KNN after PCA application
##############################################
k1_pca<-knn(train_pca, test_pca, class_train_pca, 1)
confusionMatrix(k1_pca,class_test_pca)

k3_pca<-knn(train_pca, test_pca, class_train_pca, 3)
confusionMatrix(k3_pca,class_test_pca)

k7_pca<-knn(train_pca, test_pca, class_train_pca, 7)
confusionMatrix(k7_pca,class_test_pca)

k9_pca<-knn(train_pca, test_pca, class_train_pca, 9)
confusionMatrix(k9_pca,class_test_pca)
##############################################
#SVM after PCA application
##############################################
classifier_pca = svm(formula = class~.,
                     data = train_rows_pca,
                     type = 'C-classification',
                     kernel = 'linear')

svm_pred_pca = predict(classifier_pca, newdata = test_pca)

confusionMatrix(svm_pred_pca,class_test_pca)
##############################################
#decision tree after PCA application
##############################################
model_pca <- rpart(class~., train_rows_pca, method = "class", control = rpart.control(minsplit = 1))

plot <- rpart.plot(model_pca, type = 3)

dtree_pred_pca <- predict(model_pca, test_pca, type = "class")

confusionMatrix(dtree_pred_pca,class_test_pca)
##############################################
#clustering after PCA application
##############################################
df_cluster_pca <- new_df[,-1950]

kmeans_result_pca <- kmeans(df_cluster_pca, 10)

plot3d(df_cluster, col=kmeans_result$cluster, main="k-means clusters")

fviz_nbclust(df_cluster_pca, kmeans, method = "wss")

results_pca <-table(new_df$class, kmeans_result_pca$cluster)

clusters_pca <- data_frame()
for(i in 1:10 ){
  cluster <- as.data.frame(t(results_pca[,i]))
  clusters_pca <- rbind(clusters_pca, cluster)
}

ggplot(data = clusters_pca, aes(colnames(clusters_pca),clusters_pca$`0`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 1') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters_pca$`0`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters_pca, aes(colnames(clusters_pca),clusters_pca$`1`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 2') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters_pca$`1`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters_pca, aes(colnames(clusters_pca),clusters_pca$`2`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 3') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters_pca$`2`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters_pca, aes(colnames(clusters_pca),clusters_pca$`3`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 4') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters_pca$`3`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters_pca, aes(colnames(clusters_pca),clusters_pca$`4`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 5') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters_pca$`4`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters_pca, aes(colnames(clusters_pca),clusters_pca$`5`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 6') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters_pca$`5`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters_pca, aes(colnames(clusters_pca),clusters_pca$`6`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 7') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters_pca$`6`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters_pca, aes(colnames(clusters_pca),clusters_pca$`7`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 8') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters_pca$`7`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters_pca, aes(colnames(clusters_pca),clusters_pca$`8`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 9') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters_pca$`8`), vjust=1.6, color="white", size=3.5)

ggplot(data = clusters_pca, aes(colnames(clusters_pca),clusters_pca$`9`)) + 
  geom_bar(stat = "identity", fill="steelblue") + 
  ggtitle('Quantidade de exemplares no grupo 10') +
  xlab('Exemplares') +
  ylab('Quantidade de exemplares') +
  geom_text(aes(label=clusters_pca$`9`), vjust=1.6, color="white", size=3.5)


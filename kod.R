#---------------------------------------------------------------------------------------------------------------------------
#
# ZMUM
# Projekt nr 2
#
#---------------------------------------------------------------------------------------------------------------------------
#
# Kod
#
#---------------------------------------------------------------------------------------------------------------------------

### Wczytanie pakietow

library(caret)
library(Boruta)
library(randomForest)
library(MLmetrics)
library(ggplot2)

#---------------------------------------------------------------------------------------------------------------------------

### Funkcje pomocnicze

## Funkcja do obliczania balanced accuracy (BA)
BA <- function(y_pred, y_true) {
  tab <- table(y_true, y_pred)
  balanced_accuracy <- 0.5 * (tab[1,1] / sum(tab[1,]) + tab[2,2] / sum(tab[2,]))
  balanced_accuracy
}

#---------------------------------------------------------------------------------------------------------------------------

### Wczytanie danych

train_data <- read.table("artificial_train.data")
head(train_data)
nrow(train_data)   # 2000
ncol(train_data)   # 500

train_labels <- read.table("artificial_train.labels")
head(train_labels)
nrow(train_labels)   # 2000
ncol(train_labels)   # 1
colnames(train_labels) <- "label"

valid <- read.table("artificial_valid.data")
head(valid)
nrow(valid)   # 600
ncol(valid)   # 500

#---------------------------------------------------------------------------------------------------------------------------

### Przeglad danych

table(train_labels)   # jest po 1000 obserwacji zarowno z klasy 1 jak i z klasy -1

sapply(train_data, function(x) {length(unique(x))})

#---------------------------------------------------------------------------------------------------------------------------

### Polaczenie train i train_labels w jedna ramke danych

train <- data.frame(cbind(train_data, train_labels))
head(train)
nrow(train)
ncol(train)

#---------------------------------------------------------------------------------------------------------------------------

### Zamiana labela za factor 

train$label <- as.factor(train$label)

#---------------------------------------------------------------------------------------------------------------------------

### Selekcja zmiennych + budowanie modeli + predykcja + BA

## 1) varImp ---------------------------------------------------------------------------------------------------------------

## 1.1) method = "rpart"

# Podzial na train i test
trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
data_train <- train[trainIndex,]
data_test <- train[-trainIndex,]

# Selekcja
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
model <- train(label ~ ., data = data_train, method = "rpart", preProcess = "scale", trControl = control)
importance <- varImp(model, scale = FALSE)
print(importance)

# Zmienne o importance != 0
nonzeroLabels <- rownames(importance$importance)[order(importance$importance$Overall, decreasing = TRUE)][which(importance$importance != 0)]

# Budowa modelu i predykcja
randomForestModel <- randomForest(label ~ ., data_train[,c(nonzeroLabels, "label")], ntree = 100)
randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)

# Miary
Accuracy(y_pred = randomForestPred, y_true = data_test$label)           # 0.825 ; 0.805
Precision(y_pred = randomForestPred, y_true = data_test$label)          # 0.7087379 - na wszystkich zm; 0.8095238 ; 0.7850467
BA(y_pred = randomForestPred, y_true = data_test$label)                 # 0.825 ; 0.805

# Badanie zachowania modeli zbudowanych na roznej ilosci zmiennych
acc <- prec <- ba <- numeric(length(nonzeroLabels))

for(i in 1:length(nonzeroLabels)) {
  randomForestModel <- randomForest(label ~ ., data_train[,c(nonzeroLabels[1:i], "label")], ntree = 100)
  randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)
  
  acc[i] <- Accuracy(y_pred = randomForestPred, y_true = data_test$label)
  prec[i] <- Precision(y_pred = randomForestPred, y_true = data_test$label)
  ba[i] <- BA(y_pred = randomForestPred, y_true = data_test$label)
}

acc   # 0.505 0.550 0.665 0.635 0.745 0.775 0.790 0.795
prec  # 0.5041322 0.5438596 0.6513761 0.6194690 0.7247706 0.7619048 0.7788462 0.7809524
ba    # 0.505 0.550 0.665 0.635 0.745 0.775 0.790 0.795

plt <- ggplot(data = data.frame(cbind(seq(1, length(nonzeroLabels)), ba)), aes(x = seq(1, length(nonzeroLabels)), y = ba)) +
  geom_line(color = "darkred", size = 0.9) +
  geom_point(color = "darkred", size = 4) +
  ggtitle("Balanced accuracy for random forest models") +
  xlab("Number of variables") +
  ylab("BA") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11))

# Zapis wykresu do pliku
jpeg("varImp_rpart_1.jpg")
plt
dev.off()

# Testowanie liczby drzew
n_trees <- seq(50, 500, 50)
acc <- prec <- ba <- numeric(length(n_trees))

for(j in 1:length(n_trees)) {
  randomForestModel <- randomForest(label ~ ., data_train[,c(nonzeroLabels, "label")], ntree = n_trees[j])
  randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)
  
  acc[j] <- Accuracy(y_pred = randomForestPred, y_true = data_test$label)
  prec[j] <- Precision(y_pred = randomForestPred, y_true = data_test$label)
  ba[j] <- BA(y_pred = randomForestPred, y_true = data_test$label)
}

acc   # 0.795 0.815 0.795 0.820 0.805 0.800 0.795 0.810 0.795 0.815
prec  # 0.8041237 0.8118812 0.7864078 0.7962963 0.7850467 0.7884615 0.7809524 0.7924528 0.7809524 0.8058252
ba    # 0.795 0.815 0.795 0.820 0.805 0.800 0.795 0.810 0.795 0.815


# Petla - 10 iteracji
number_of_iterations <- 10
acc <- prec <- ba <- matrix(0, nrow = number_of_iterations, ncol = length(nonzeroLabels))
modelsNames <- matrix(nrow = number_of_iterations, ncol = length(nonzeroLabels))

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # Selekcja
  control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  model <- train(label ~ ., data = data_train, method = "rpart", preProcess = "scale", trControl = control)
  importance <- varImp(model, scale = FALSE)
  
  # Zmienne o importance != 0
  nonzeroLabels <- rownames(importance$importance)[order(importance$importance$Overall, decreasing = TRUE)][which(importance$importance != 0)]
  
  j <- 1
  
  for(j in 1:length(nonzeroLabels)) {
    randomForestModel <- randomForest(label ~ ., data_train[,c(nonzeroLabels[1:j], "label")], ntree = 100)
    randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
    randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)
    
    modelsNames[i, j] <- paste0(nonzeroLabels[1:j][order(nonzeroLabels[1:j])], collapse = " + ")
    acc[i, j] <- Accuracy(y_pred = randomForestPred, y_true = data_test$label)
    prec[i, j] <- Precision(y_pred = randomForestPred, y_true = data_test$label)
    ba[i, j] <- BA(y_pred = randomForestPred, y_true = data_test$label)
  }
}


baMeans <- matrix(0, nrow = ncol(modelsNames), ncol = nrow(modelsNames))

for(i in 1:ncol(modelsNames)) {
  uniques <- unique(modelsNames[,i])
  
  j <- 1
  
  for(j in 1:length(uniques)) {
    idx <- which(modelsNames[,i] == unique(modelsNames[,i])[j])
    baMeans[i, j] <- mean(ba[idx, i], na.rm = TRUE)
  }
}

# Najlepszy model (wg BA)
max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))        # 0.865
which.max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))  # 9
which.max(baMeans[7,])                                                     # 3
modelsNames[3, 9]                                                          # "V106 + V242 + V29 + V319 + V337 + V339 + V452 + V473 + V476"

# Wykres BA od modelu (maksima po liczbie zmiennych w modelu srednich dla modeli)
y <- apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)})[1:9]

plt2 <- ggplot(data = data.frame(cbind(seq(1, 9), y)), aes(x = seq(1, 9), y = y)) +
  geom_line(color = "darkred", size = 0.9) +
  geom_point(color = "darkred", size = 4) +
  scale_x_discrete(limits=c(seq(1, 9)))+
  ggtitle("Balanced accuracy for random forest models") +
  xlab("Number of variables") +
  ylab("BA") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11))

plt2

# Zapis wykresu do pliku
jpeg("varImp_rpart_2.jpg")
plt2
dev.off()


# Petla - 20 iteracji
number_of_iterations <- 20
acc <- prec <- ba <- matrix(0, nrow = number_of_iterations, ncol = 15)
modelsNames <- matrix(nrow = number_of_iterations, ncol = 15)

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # Selekcja
  control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  model <- train(label ~ ., data = data_train, method = "rpart", preProcess = "scale", trControl = control)
  importance <- varImp(model, scale = FALSE)
  
  # Zmienne o importance != 0
  nonzeroLabels <- rownames(importance$importance)[order(importance$importance$Overall, decreasing = TRUE)][which(importance$importance != 0)]
  
  j <- 1
  
  for(j in 1:length(nonzeroLabels)) {
    randomForestModel <- randomForest(label ~ ., data_train[,c(nonzeroLabels[1:j], "label")], ntree = 100)
    randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
    randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)
    
    modelsNames[i, j] <- paste0(nonzeroLabels[1:j][order(nonzeroLabels[1:j])], collapse = " + ")
    acc[i, j] <- Accuracy(y_pred = randomForestPred, y_true = data_test$label)
    prec[i, j] <- Precision(y_pred = randomForestPred, y_true = data_test$label)
    ba[i, j] <- BA(y_pred = randomForestPred, y_true = data_test$label)
  }
}


baMeans <- matrix(0, nrow = ncol(modelsNames), ncol = nrow(modelsNames))

for(i in 1:ncol(modelsNames)) {
  uniques <- unique(modelsNames[,i])
  
  j <- 1
  
  for(j in 1:length(uniques)) {
    idx <- which(modelsNames[,i] == unique(modelsNames[,i])[j])
    baMeans[i, j] <- mean(ba[idx, i], na.rm = TRUE)
  }
}

# Najlepszy model (wg BA)
max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))        # 0.87
which.max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))  # 8
which.max(baMeans[8,])                                                     # 10
modelsNames[10, 8]                                                         # "V106 + V129 + V242 + V337 + V339 + V379 + V476 + V49"

# Wykres BA od modelu (maksima po liczbie zmiennych w modelu srednich dla modeli)
y <- apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)})[1:9]

plt3 <- ggplot(data = data.frame(cbind(seq(1, 9), y)), aes(x = seq(1, 9), y = y)) +
  geom_line(color = "darkred", size = 0.9) +
  geom_point(color = "darkred", size = 4) +
  scale_x_discrete(limits=c(seq(1, 9)))+
  ggtitle("Balanced accuracy for random forest models") +
  xlab("Number of variables") +
  ylab("BA") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11))

plt3

# Zapis wykresu do pliku
jpeg("varImp_rpart_3.jpg")
plt3
dev.off()


# Test dla najlepszego modeli w celu lepszego usrednienia miar
number_of_iterations <- 2000
acc8 <- prec8 <- ba8 <- numeric(number_of_iterations)

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # 8 zmiennych wersja 1
  randomForestModel8 <- randomForest(label ~ ., data_train[,c("V106", "V129", "V242", "V337", "V339", "V379", "V476", "V49", "label")], ntree = 100)
  randomForestProb8 <- predict(randomForestModel8, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred8 <- ifelse(randomForestProb8 < 0.5, -1, 1)
  
  acc8[i] <- Accuracy(y_pred = randomForestPred8, y_true = data_test$label)
  prec8[i] <- Precision(y_pred = randomForestPred8, y_true = data_test$label)
  ba8[i] <- BA(y_pred = randomForestPred8, y_true = data_test$label)
}

# 50 iter   ; 100 iter  ; 2000 iteracji
mean(acc8)   # 0.7972    ; 0.797     ; 0.8017375
mean(prec8)  # 0.7912516 ; 0.7937188 ; 0.795389
mean(ba8)    # 0.7972    ; 0.797     ; 0.8017375



## 1.2) method = "lvq"

# Podzial na train i test
trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
data_train <- train[trainIndex,]
data_test <- train[-trainIndex,]

# Selekcja
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
model <- train(label ~ ., data = data_train, method = "lvq", preProcess = "scale", trControl = control)
importance <- varImp(model, scale = FALSE)
print(importance)

# Zmienne o importance != 0
nonzeroLabels <- rownames(importance$importance)[order(importance$importance$X1, decreasing = TRUE)][which(importance$importance$X1 != 0)]

# Budowa modelu i predykcja
randomForestModel <- randomForest(label ~ ., data_train[,c(nonzeroLabels, "label")], ntree = 100)
randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)

# Miary
Accuracy(y_pred = randomForestPred, y_true = data_test$label)           # 0.64
Precision(y_pred = randomForestPred, y_true = data_test$label)          # 0.6320755
BA(y_pred = randomForestPred, y_true = data_test$label)                 # 0.64

# Badanie zachowania modeli zbudowanych na roznej ilosci zmiennych
acc <- prec <- ba <- numeric(20)

for(i in 1:20) {
  randomForestModel <- randomForest(label ~ ., data_train[,c(nonzeroLabels[1:i], "label")], ntree = 100)
  randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)
  
  acc[i] <- Accuracy(y_pred = randomForestPred, y_true = data_test$label)
  prec[i] <- Precision(y_pred = randomForestPred, y_true = data_test$label)
  ba[i] <- BA(y_pred = randomForestPred, y_true = data_test$label)
}

acc   # 0.585 0.560 0.555 0.555 0.680 0.725 0.750 0.825 0.845 0.850 0.840 0.895 0.890 0.870 0.890 0.880 0.875 0.870 0.850 0.835
prec  # 0.5841584 0.5638298 0.5591398 0.5567010 0.6914894 0.7227723 0.7403846 0.7981651 0.8165138 0.8365385 0.8090909
# 0.8761905 0.8823529 0.8627451 0.8823529 0.8518519 0.8571429 0.8700000 0.8181818 0.8073394
ba    # 0.585 0.560 0.555 0.555 0.680 0.725 0.750 0.825 0.845 0.850 0.840 0.895 0.890 0.870 0.890 0.880 0.875 0.870 0.850 0.835

plt <- ggplot(data = data.frame(cbind(seq(1, 20), ba)), aes(x = seq(1, 20), y = ba)) +
  geom_line(color = "darkred", size = 0.9) +
  geom_point(color = "darkred", size = 4) +
  ggtitle("Balanced accuracy for random forest models") +
  xlab("Number of variables") +
  ylab("BA") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11))

plt

# Zapis wykresu do pliku
jpeg("varImp_lvq_1.jpg")
plt
dev.off()

# Petla - 5 iteracji
number_of_iterations <- 5
acc <- prec <- ba <- matrix(0, nrow = number_of_iterations, ncol = 20)
modelsNames <- matrix(nrow = number_of_iterations, ncol = 20)

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # Selekcja
  control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  model <- train(label ~ ., data = data_train, method = "lvq", preProcess = "scale", trControl = control)
  importance <- varImp(model, scale = FALSE)
  
  # Zmienne o importance != 0
  nonzeroLabels <- rownames(importance$importance)[order(importance$importance$X1, decreasing = TRUE)][which(importance$importance$X1 != 0)]
  
  j <- 1
  
  for(j in 1:20) {
    randomForestModel <- randomForest(label ~ ., data_train[,c(nonzeroLabels[1:j], "label")], ntree = 100)
    randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
    randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)
    
    modelsNames[i, j] <- paste0(nonzeroLabels[1:j][order(nonzeroLabels[1:j])], collapse = " + ")
    acc[i, j] <- Accuracy(y_pred = randomForestPred, y_true = data_test$label)
    prec[i, j] <- Precision(y_pred = randomForestPred, y_true = data_test$label)
    ba[i, j] <- BA(y_pred = randomForestPred, y_true = data_test$label)
  }
}


baMeans <- matrix(0, nrow = ncol(modelsNames), ncol = nrow(modelsNames))

for(i in 1:ncol(modelsNames)) {
  uniques <- unique(modelsNames[,i])
  
  j <- 1
  
  for(j in 1:length(uniques)) {
    idx <- which(modelsNames[,i] == unique(modelsNames[,i])[j])
    baMeans[i, j] <- mean(ba[idx, i], na.rm = TRUE)
  }
}

# Najlepszy model (wg BA)
max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))        # 0.895
which.max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))  # 15
which.max(baMeans[15,])                                                    # 2
modelsNames[2, 15]                                                          
# "V106 + V129 + V137 + V242 + V337 + V339 + V379 + V443 + V454 + V473 + V476 + V49 + V494 + V5 + V65"

# Wykres BA od modelu (maksima po liczbie zmiennych w modelu srednich dla modeli)
y <- apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)})[1:20]

plt2 <- ggplot(data = data.frame(cbind(seq(1, 20), y)), aes(x = seq(1, 20), y = y)) +
  geom_line(color = "darkred", size = 0.9) +
  geom_point(color = "darkred", size = 4) +
  scale_x_discrete(limits=c(seq(1, 20)))+
  ggtitle("Balanced accuracy for random forest models") +
  xlab("Number of variables") +
  ylab("BA") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11))

plt2

# Zapis wykresu do pliku
jpeg("varImp_lvq_2.jpg")
plt2
dev.off()


# Test dla dwoch najlepszych modeli w celu lepszego usrednienia miar
number_of_iterations <- 2000
acc12 <- prec12 <- ba12 <- acc15 <- prec15 <- ba15 <- numeric(number_of_iterations)

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # 12 zmiennych
  randomForestModel12 <- randomForest(label ~ ., data_train[,c("V106", "V129", "V242", "V337", "V339", "V379", "V443", "V454", "V473", "V476", "V49", "V65", "label")], ntree = 100)
  randomForestProb12 <- predict(randomForestModel12, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred12 <- ifelse(randomForestProb12 < 0.5, -1, 1)
  
  acc12[i] <- Accuracy(y_pred = randomForestPred12, y_true = data_test$label)
  prec12[i] <- Precision(y_pred = randomForestPred12, y_true = data_test$label)
  ba12[i] <- BA(y_pred = randomForestPred12, y_true = data_test$label)
  
  # 15 zmiennych
  randomForestModel15 <- randomForest(label ~ ., data_train[,c("V106", "V129", "V137", "V242", "V337", "V339", "V379", "V443", "V454", "V473", "V476", "V49", "V494", "V5", "V65", "label")], ntree = 100)
  randomForestProb15 <- predict(randomForestModel15, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred15 <- ifelse(randomForestProb15 < 0.5, -1, 1)
  
  acc15[i] <- Accuracy(y_pred = randomForestPred15, y_true = data_test$label)
  prec15[i] <- Precision(y_pred = randomForestPred15, y_true = data_test$label)
  ba15[i] <- BA(y_pred = randomForestPred15, y_true = data_test$label)
}

# 2000 iteracji
mean(acc12)   # 0.8690075
mean(prec12)  # 0.8686728
mean(ba12)    # 0.8690075

mean(acc15)   # 0.8690725
mean(prec15)  # 0.8666718
mean(ba15)    # 0.8690725


# Test dla najlepszego modelu w celu lepszego usrednienia miar
number_of_iterations <- 2000
acc15 <- prec15 <- ba15 <- numeric(number_of_iterations)

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # 15 zmiennych
  randomForestModel15 <- randomForest(label ~ ., data_train[,c("V106", "V129", "V137", "V242", "V337", "V339", "V379", "V443", "V454", "V473", "V476", "V49", "V494", "V5", "V65", "label")], ntree = 100)
  randomForestProb15 <- predict(randomForestModel15, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred15 <- ifelse(randomForestProb15 < 0.5, -1, 1)
  
  acc15[i] <- Accuracy(y_pred = randomForestPred15, y_true = data_test$label)
  prec15[i] <- Precision(y_pred = randomForestPred15, y_true = data_test$label)
  ba15[i] <- BA(y_pred = randomForestPred15, y_true = data_test$label)
}

# 2000 iteracji
mean(acc15)   # 0.869
mean(prec15)  # 0.8665179
mean(ba15)    # 0.869




## 1.3) method = "rf"

# Podzial na train i test
trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
data_train <- train[trainIndex,]
data_test <- train[-trainIndex,]

# Selekcja
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
model <- train(label ~ ., data = data_train, method = "rf", preProcess = "scale", trControl = control)
importance <- varImp(model, scale = FALSE)
print(importance)

# Zmienne o importance != 0
nonzeroLabels <- rownames(importance$importance)[order(importance$importance$Overall, decreasing = TRUE)][which(importance$importance != 0)]

# Budowa modelu i predykcja
randomForestModel <- randomForest(label ~ ., data_train[,c(nonzeroLabels, "label")], ntree = 100)
randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)

# Miary
Accuracy(y_pred = randomForestPred, y_true = data_test$label)           # 0.685
Precision(y_pred = randomForestPred, y_true = data_test$label)          # 0.6796117
BA(y_pred = randomForestPred, y_true = data_test$label)                 # 0.685

# Badanie zachowania modeli zbudowanych na roznej ilosci zmiennych
acc <- prec <- ba <- numeric(20)

for(i in 1:20) {
  randomForestModel <- randomForest(label ~ ., data_train[,c(nonzeroLabels[1:i], "label")], ntree = 100)
  randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)
  
  acc[i] <- Accuracy(y_pred = randomForestPred, y_true = data_test$label)
  prec[i] <- Precision(y_pred = randomForestPred, y_true = data_test$label)
  ba[i] <- BA(y_pred = randomForestPred, y_true = data_test$label)
}

acc   # 0.590 0.610 0.695 0.760 0.760 0.830 0.845 0.860 0.875 0.870 0.875 0.870 0.880 0.850 0.875 0.860 0.885 0.885 0.885 0.885
prec  # 0.5803571 0.6000000 0.6857143 0.7452830 0.7500000 0.8113208 0.8224299 0.8333333 0.8504673 0.8700000
# 0.8865979 0.8627451 0.8725490 0.8431373 0.8640777 0.8529412 0.8737864 0.8811881 0.8811881 0.8811881
ba    # 0.590 0.610 0.695 0.760 0.760 0.830 0.845 0.860 0.875 0.870 0.875 0.870 0.880 0.850 0.875 0.860 0.885 0.885 0.885 0.885

plt <- ggplot(data = data.frame(cbind(seq(1, 20), ba)), aes(x = seq(1, 20), y = ba)) +
  geom_line(color = "darkred", size = 0.9) +
  geom_point(color = "darkred", size = 4) +
  ggtitle("Balanced accuracy for random forest models") +
  xlab("Number of variables") +
  ylab("BA") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11))

plt

# Zapis wykresu do pliku
jpeg("varImp_rf_1.jpg")
plt
dev.off()

# Petla - 5 iteracji
number_of_iterations <- 5
acc <- prec <- ba <- matrix(0, nrow = number_of_iterations, ncol = 20)
modelsNames <- matrix(nrow = number_of_iterations, ncol = 20)

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # Selekcja
  control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  model <- train(label ~ ., data = data_train, method = "rf", preProcess = "scale", trControl = control)
  importance <- varImp(model, scale = FALSE)
  
  # Zmienne o importance != 0
  nonzeroLabels <- rownames(importance$importance)[order(importance$importance$Overall, decreasing = TRUE)][which(importance$importance != 0)]
  
  j <- 1
  
  for(j in 1:20) {
    randomForestModel <- randomForest(label ~ ., data_train[,c(nonzeroLabels[1:j], "label")], ntree = 100)
    randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
    randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)
    
    modelsNames[i, j] <- paste0(nonzeroLabels[1:j][order(nonzeroLabels[1:j])], collapse = " + ")
    acc[i, j] <- Accuracy(y_pred = randomForestPred, y_true = data_test$label)
    prec[i, j] <- Precision(y_pred = randomForestPred, y_true = data_test$label)
    ba[i, j] <- BA(y_pred = randomForestPred, y_true = data_test$label)
  }
}


baMeans <- matrix(0, nrow = ncol(modelsNames), ncol = nrow(modelsNames))

for(i in 1:ncol(modelsNames)) {
  uniques <- unique(modelsNames[,i])
  
  j <- 1
  
  for(j in 1:length(uniques)) {
    idx <- which(modelsNames[,i] == unique(modelsNames[,i])[j])
    baMeans[i, j] <- mean(ba[idx, i], na.rm = TRUE)
  }
}

# Najlepszy model (wg BA)
max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))        # 0.915
which.max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))  # 15
which.max(baMeans[15,])                                                    # 1
modelsNames[1, 15]                                                          
# "V106 + V154 + V242 + V282 + V29 + V319 + V339 + V379 + V434 + V443 + V454 + V473 + V476 + V49 + V494"

# Wykres BA od modelu (maksima po liczbie zmiennych w modelu srednich dla modeli)
y <- apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)})[1:20]

plt2 <- ggplot(data = data.frame(cbind(seq(1, 20), y)), aes(x = seq(1, 20), y = y)) +
  geom_line(color = "darkred", size = 0.9) +
  geom_point(color = "darkred", size = 4) +
  scale_x_discrete(limits=c(seq(1, 20)))+
  ggtitle("Balanced accuracy for random forest models") +
  xlab("Number of variables") +
  ylab("BA") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11))

plt2

# Zapis wykresu do pliku
jpeg("varImp_rf_2.jpg")
plt2
dev.off()


# Test dla dwoch najlepszych modeli w celu lepszego usrednienia miar
number_of_iterations <- 2000
acc81 <- prec81 <- ba81 <- acc82 <- prec82 <- ba82 <- acc15 <- prec15 <- ba15 <- numeric(number_of_iterations)

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # 8 zmiennych wersja 1
  randomForestModel81 <- randomForest(label ~ ., data_train[,c("V106", "V154", "V319", "V339", "V379", "V443", "V476", "V49", "label")], ntree = 100)
  randomForestProb81 <- predict(randomForestModel81, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred81 <- ifelse(randomForestProb81 < 0.5, -1, 1)
  
  acc81[i] <- Accuracy(y_pred = randomForestPred81, y_true = data_test$label)
  prec81[i] <- Precision(y_pred = randomForestPred81, y_true = data_test$label)
  ba81[i] <- BA(y_pred = randomForestPred81, y_true = data_test$label)
  
  # 8 zmiennych wersja 2
  randomForestModel82 <- randomForest(label ~ ., data_train[,c("V106", "V242", "V319", "V339", "V379", "V443", "V476", "V49", "label")], ntree = 100)
  randomForestProb82 <- predict(randomForestModel82, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred82 <- ifelse(randomForestProb82 < 0.5, -1, 1)
  
  acc82[i] <- Accuracy(y_pred = randomForestPred82, y_true = data_test$label)
  prec82[i] <- Precision(y_pred = randomForestPred82, y_true = data_test$label)
  ba82[i] <- BA(y_pred = randomForestPred82, y_true = data_test$label)
  
  # 15 zmiennych
  randomForestModel15 <- randomForest(label ~ ., data_train[,c("V106", "V154", "V242", "V282", "V29", "V319", "V339", "V379", "V434", "V443", "V454", "V473", "V476", "V49", "V494", "label")], ntree = 100)
  randomForestProb15 <- predict(randomForestModel15, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred15 <- ifelse(randomForestProb15 < 0.5, -1, 1)
  
  acc15[i] <- Accuracy(y_pred = randomForestPred15, y_true = data_test$label)
  prec15[i] <- Precision(y_pred = randomForestPred15, y_true = data_test$label)
  ba15[i] <- BA(y_pred = randomForestPred15, y_true = data_test$label)
}

# 50 iter   ; 100 iter  ; 2000 iteracji

mean(acc81)   # 0.8773    ; 0.8766    ; 0.8741725
mean(prec81)  # 0.8779827 ; 0.873518  ; 0.8709384
mean(ba81)    # 0.8773    ; 0.8766    ; 0.8741725

mean(acc82)   # 0.859     ; 0.85825   ; 0.8581125
mean(prec82)  # 0.8540879 ; 0.8570264 ; 0.852271
mean(ba82)    # 0.859     ; 0.85825   ; 0.8581125

mean(acc15)   # 0.8796    ; 0.8796    ; 0.87855
mean(prec15)  # 0.8827857 ; 0.8808262 ; 0.8772902
mean(ba15)    # 0.8796    ; 0.8796    ; 0.87855



## 2) Boruta ---------------------------------------------------------------------------------------------------

boruta_output <- Boruta(label ~ ., data = data_train, doTrace = 0)  
roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

imps <- attStats(roughFixMod)
imps2 = imps[imps$decision != "Rejected", c("meanImp", "decision")]
head(imps2[order(-imps2$meanImp), ])

plot(boruta_output, cex.axis = .7, las = 2, xlab = "", main = "Variable Importance") 

lbls <- rownames(imps2[order(-imps2$meanImp), ])

# Budowa modelu i predykcja
randomForestModel <- randomForest(label ~ ., data_train[,c(lbls, "label")], ntree = 100)
randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)

# Miary
Accuracy(y_pred = randomForestPred, y_true = data_test$label)           # 0.89
Precision(y_pred = randomForestPred, y_true = data_test$label)          # 0.89
BA(y_pred = randomForestPred, y_true = data_test$label)                 # 0.89

# Badanie zachowania modeli zbudowanych na roznej ilosci zmiennych
acc <- prec <- ba <- numeric(length(lbls))

for(i in 1:length(lbls)) {
  randomForestModel <- randomForest(label ~ ., data_train[,c(lbls[1:i], "label")], ntree = 100)
  randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)
  
  acc[i] <- Accuracy(y_pred = randomForestPred, y_true = data_test$label)
  prec[i] <- Precision(y_pred = randomForestPred, y_true = data_test$label)
  ba[i] <- BA(y_pred = randomForestPred, y_true = data_test$label)
}

acc   # 0.510 0.625 0.610 0.685 0.720 0.805 0.845 0.855 0.865 0.870 0.870 0.865 0.880 0.885 0.880 0.885 0.880 0.890 0.885 0.880
prec  # 0.5098039 0.6237624 0.6145833 0.6868687 0.7340426 0.8080808 0.8556701 0.8659794 0.8613861 0.8700000
# 0.8700000 0.8686869 0.8725490 0.8888889 0.8800000 0.8811881 0.8725490 0.8979592 0.8888889 0.8800000
ba    # 0.510 0.625 0.610 0.685 0.720 0.805 0.845 0.855 0.865 0.870 0.870 0.865 0.880 0.885 0.880 0.885 0.880 0.890 0.885 0.880

plt <- ggplot(data = data.frame(cbind(seq(1, length(lbls)), ba)), aes(x = seq(1, length(lbls)), y = ba)) +
  geom_line(color = "darkred", size = 0.9) +
  geom_point(color = "darkred", size = 4) +
  ggtitle("Balanced accuracy for random forest models") +
  xlab("Number of variables") +
  ylab("BA") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11))

plt

# Zapis wykresu do pliku
jpeg("boruta_1.jpg")
plt
dev.off()


# Petla - 10 iteracji
number_of_iterations <- 10
acc <- prec <- ba <- matrix(0, nrow = number_of_iterations, ncol = 25)
modelsNames <- matrix(nrow = number_of_iterations, ncol = 25)

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # Selekcja
  boruta_output <- Boruta(label ~ ., data = data_train, doTrace = 0)  
  roughFixMod <- TentativeRoughFix(boruta_output)
  boruta_signif <- getSelectedAttributes(roughFixMod)
  imps <- attStats(roughFixMod)
  imps2 = imps[imps$decision != "Rejected", c("meanImp", "decision")]
  lbls <- rownames(imps2[order(-imps2$meanImp), ])
  
  j <- 1
  
  for(j in 1:length(lbls)) {
    randomForestModel <- randomForest(label ~ ., data_train[,c(lbls[1:j], "label")], ntree = 100)
    randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
    randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)
    
    modelsNames[i, j] <- paste0(lbls[1:j][order(lbls[1:j])], collapse = " + ")
    acc[i, j] <- Accuracy(y_pred = randomForestPred, y_true = data_test$label)
    prec[i, j] <- Precision(y_pred = randomForestPred, y_true = data_test$label)
    ba[i, j] <- BA(y_pred = randomForestPred, y_true = data_test$label)
  }
}


baMeans <- matrix(0, nrow = ncol(modelsNames), ncol = nrow(modelsNames))

for(i in 1:ncol(modelsNames)) {
  uniques <- unique(modelsNames[,i])
  
  j <- 1
  
  for(j in 1:length(uniques)) {
    idx <- which(modelsNames[,i] == unique(modelsNames[,i])[j])
    baMeans[i, j] <- mean(ba[idx, i], na.rm = TRUE)
  }
}

# Najlepszy model (wg BA)
max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))        # 0.925
which.max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))  # 16
which.max(baMeans[16,])                                                    # 3
modelsNames[3, 16]                                                         # "V106 + V129 + V154 + V242 + V282 + V29 + V319 + V337 + V339 + V379 + V434 + V443 + V452 + V473 + V476 + V49"

# Wykres BA od modelu (maksima po liczbie zmiennych w modelu srednich dla modeli)
y <- apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)})[1:21]

plt2 <- ggplot(data = data.frame(cbind(seq(1, 21), y)), aes(x = seq(1, 21), y = y)) +
  geom_line(color = "darkred", size = 0.9) +
  geom_point(color = "darkred", size = 4) +
  scale_x_discrete(limits=c(seq(1, 21)))+
  ggtitle("Balanced accuracy for random forest models") +
  xlab("Number of variables") +
  ylab("BA") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11))

plt2

# Zapis wykresu do pliku
jpeg("boruta_2.jpg")
plt2
dev.off()


# Petla - 30 iteracji
number_of_iterations <- 30
acc <- prec <- ba <- matrix(0, nrow = number_of_iterations, ncol = 25)
modelsNames <- matrix(nrow = number_of_iterations, ncol = 25)

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # Selekcja
  boruta_output <- Boruta(label ~ ., data = data_train, doTrace = 0)  
  roughFixMod <- TentativeRoughFix(boruta_output)
  boruta_signif <- getSelectedAttributes(roughFixMod)
  imps <- attStats(roughFixMod)
  imps2 = imps[imps$decision != "Rejected", c("meanImp", "decision")]
  lbls <- rownames(imps2[order(-imps2$meanImp), ])
  
  j <- 1
  
  for(j in 1:length(lbls)) {
    randomForestModel <- randomForest(label ~ ., data_train[,c(lbls[1:j], "label")], ntree = 100)
    randomForestProb <- predict(randomForestModel, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
    randomForestPred <- ifelse(randomForestProb < 0.5, -1, 1)
    
    modelsNames[i, j] <- paste0(lbls[1:j][order(lbls[1:j])], collapse = " + ")
    acc[i, j] <- Accuracy(y_pred = randomForestPred, y_true = data_test$label)
    prec[i, j] <- Precision(y_pred = randomForestPred, y_true = data_test$label)
    ba[i, j] <- BA(y_pred = randomForestPred, y_true = data_test$label)
  }
}


baMeans <- matrix(0, nrow = ncol(modelsNames), ncol = nrow(modelsNames))

for(i in 1:ncol(modelsNames)) {
  uniques <- unique(modelsNames[,i])
  
  j <- 1
  
  for(j in 1:length(uniques)) {
    idx <- which(modelsNames[,i] == unique(modelsNames[,i])[j])
    baMeans[i, j] <- mean(ba[idx, i], na.rm = TRUE)
  }
}

# Najlepszy model (wg BA)
max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))        # 0.92
which.max(apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)}))  # 20
which.max(baMeans[20,])                                                    # 3
modelsNames[3, 20]                                                         
# "V106 + V129 + V154 + V242 + V282 + V29 + V319 + V337 + V339 + V379 + V434 + V443 + V452 + V454 + V456 + V473 + V476 + V49 + V494 + V65"

# Wykres BA od modelu (maksima po liczbie zmiennych w modelu srednich dla modeli)
y <- apply(baMeans, MARGIN = 1, function(x) {max(x, na.rm = TRUE)})[1:21]

plt3 <- ggplot(data = data.frame(cbind(seq(1, 21), y)), aes(x = seq(1, 21), y = y)) +
  geom_line(color = "darkred", size = 0.9) +
  geom_point(color = "darkred", size = 4) +
  scale_x_discrete(limits=c(seq(1, 21)))+
  ggtitle("Balanced accuracy for random forest models") +
  xlab("Number of variables") +
  ylab("BA") +
  theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text = element_text(size = 11))

plt3

# Zapis wykresu do pliku
jpeg("boruta_3.jpg")
plt3
dev.off()




# Test dla dwoch najlepszych modeli w celu lepszego usrednienia miar
number_of_iterations <- 2000
acc16 <- prec16 <- ba16 <- acc20 <- prec20 <- ba20 <- numeric(number_of_iterations)

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # 16 zmiennych
  randomForestModel16 <- randomForest(label ~ ., data_train[,c("V106", "V129", "V154", "V242", "V282", "V29", "V319", "V337", "V339", "V379", "V434", "V443", "V452", "V473", "V476", "V49", "label")], ntree = 100)
  randomForestProb16 <- predict(randomForestModel16, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred16 <- ifelse(randomForestProb16 < 0.5, -1, 1)
  
  acc16[i] <- Accuracy(y_pred = randomForestPred16, y_true = data_test$label)
  prec16[i] <- Precision(y_pred = randomForestPred16, y_true = data_test$label)
  ba16[i] <- BA(y_pred = randomForestPred16, y_true = data_test$label)
  
  # 20 zmiennych
  randomForestModel20 <- randomForest(label ~ ., data_train[,c("V106", "V129", "V154", "V242", "V282", "V29", "V319", "V337", "V339", "V379", "V434", "V443", "V452", "V454", "V456", "V473", "V476", "V49", "V494", "V65", "label")], ntree = 100)
  randomForestProb20 <- predict(randomForestModel20, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred20 <- ifelse(randomForestProb20 < 0.5, -1, 1)
  
  acc20[i] <- Accuracy(y_pred = randomForestPred20, y_true = data_test$label)
  prec20[i] <- Precision(y_pred = randomForestPred20, y_true = data_test$label)
  ba20[i] <- BA(y_pred = randomForestPred20, y_true = data_test$label)
}

# 50 iter   ; 100 iter  ; 2000 iteracji
mean(acc16)   # 0.8787    ; 0.88485   ; 0.8810025
mean(prec16)  # 0.8772008 ; 0.8872107 ; 0.8814183
mean(ba16)    # 0.8787    ; 0.88485   ; 0.8810025

mean(acc20)   # 0.8849    ; 0.8896    ; 0.8876225
mean(prec20)  # 0.8828944 ; 0.8907385 ; 0.886605
mean(ba20)    # 0.8849    ; 0.8896    ; 0.8876225


# Test dla dwoch najlepszych modeli w celu lepszego usrednienia miar
number_of_iterations <- 2000
acc161 <- prec161 <- ba161 <- acc162 <- prec162 <- ba162 <- acc163 <- prec163 <- ba163 <- numeric(number_of_iterations)
acc201 <- prec201 <- ba201 <- acc202 <- prec202 <- ba202 <- numeric(number_of_iterations)

for(i in 1:number_of_iterations) {
  # Podzial na train i test
  trainIndex <- createDataPartition(train$label, p = 0.9, list = FALSE)
  data_train <- train[trainIndex,]
  data_test <- train[-trainIndex,]
  
  # 16 zmiennych wersja 1
  randomForestModel161 <- randomForest(label ~ ., data_train[,c("V106", "V129", "V154", "V242", "V282", "V29", "V319", "V337", "V339", "V379", "V434", "V443", "V452", "V473", "V476", "V49", "label")], ntree = 100)
  randomForestProb161 <- predict(randomForestModel161, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred161 <- ifelse(randomForestProb161 < 0.5, -1, 1)
  
  acc161[i] <- Accuracy(y_pred = randomForestPred161, y_true = data_test$label)
  prec161[i] <- Precision(y_pred = randomForestPred161, y_true = data_test$label)
  ba161[i] <- BA(y_pred = randomForestPred161, y_true = data_test$label)
  
  # 16 zmiennych wersja 2
  randomForestModel162 <- randomForest(label ~ ., data_train[,c("V106", "V129", "V154", "V242", "V282", "V29", "V319", "V337", "V339", "V379", "V434", "V443", "V452", "V473", "V476", "V49", "label")], ntree = 100)
  randomForestProb162 <- predict(randomForestModel162, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred162 <- ifelse(randomForestProb162 < 0.5, -1, 1)
  
  acc162[i] <- Accuracy(y_pred = randomForestPred162, y_true = data_test$label)
  prec162[i] <- Precision(y_pred = randomForestPred162, y_true = data_test$label)
  ba162[i] <- BA(y_pred = randomForestPred162, y_true = data_test$label)
  
  # 16 zmiennych wersja 3
  randomForestModel163 <- randomForest(label ~ ., data_train[,c("V106", "V129", "V154", "V242", "V282", "V29", "V319", "V337", "V339", "V379", "V434", "V443", "V452", "V473", "V476", "V49", "label")], ntree = 100)
  randomForestProb163 <- predict(randomForestModel163, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred163 <- ifelse(randomForestProb163 < 0.5, -1, 1)
  
  acc163[i] <- Accuracy(y_pred = randomForestPred163, y_true = data_test$label)
  prec163[i] <- Precision(y_pred = randomForestPred163, y_true = data_test$label)
  ba163[i] <- BA(y_pred = randomForestPred163, y_true = data_test$label)
  
  # 20 zmiennych wersja 1
  randomForestModel201 <- randomForest(label ~ ., data_train[,c("V106", "V129", "V154", "V242", "V282", "V29", "V319", "V337", "V339", "V379", "V434", "V443", "V452", "V454", "V456", "V473", "V476", "V49", "V494", "V65", "label")], ntree = 100)
  randomForestProb201 <- predict(randomForestModel201, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred201 <- ifelse(randomForestProb201 < 0.5, -1, 1)
  
  acc201[i] <- Accuracy(y_pred = randomForestPred201, y_true = data_test$label)
  prec201[i] <- Precision(y_pred = randomForestPred201, y_true = data_test$label)
  ba201[i] <- BA(y_pred = randomForestPred201, y_true = data_test$label)
  
  # 20 zmiennych wersja 2
  randomForestModel202 <- randomForest(label ~ ., data_train[,c("V106", "V129", "V154", "V242", "V282", "V29", "V319", "V337", "V339", "V379", "V432", "V434", "V443", "V452", "V454", "V473", "V476", "V49", "V494", "V65", "label")], ntree = 100)
  randomForestProb202 <- predict(randomForestModel202, newdata = data_test[,-ncol(data_test)], type = 'prob')[,2]
  randomForestPred202 <- ifelse(randomForestProb202 < 0.5, -1, 1)
  
  acc202[i] <- Accuracy(y_pred = randomForestPred202, y_true = data_test$label)
  prec202[i] <- Precision(y_pred = randomForestPred202, y_true = data_test$label)
  ba202[i] <- BA(y_pred = randomForestPred202, y_true = data_test$label)
}

# 2000 iteracji
mean(acc161)   # 0.8827325
mean(prec161)  # 0.8846376
mean(ba161)    # 0.8827325

mean(acc162)   # 0.882755
mean(prec162)  # 0.8846474
mean(ba162)    # 0.882755

mean(acc163)   # 0.8828825
mean(prec163)  # 0.8847616
mean(ba163)    # 0.8828825

mean(acc201)   # 0.88942
mean(prec201)  # 0.8896646
mean(ba201)    # 0.88942

mean(acc202)   # 0.8864775
mean(prec202)  # 0.8861936
mean(ba202)    # 0.8864775


#---------------------------------------------------------------------------------------------------------------------------

# Ostatecznie wybra³am metodê Boruta z 16 zmiennymi
# V106 + V129 + V154 + V242 + V282 + V29 + V319 + V337 + V339 + V379 + V434 + V443 + V452 + V473 + V476 + V49

#---------------------------------------------------------------------------------------------------------------------------

### Predykcja dla danych ze zbioru artificial_valid.data

# Wczytanie danych
train_data <- read.table("artificial_train.data")
train_labels <- read.table("artificial_train.labels")
valid <- read.table("artificial_valid.data")

# Polaczenie train i train_labels w jedna ramke danych
colnames(train_labels) <- "label"
train <- data.frame(cbind(train_data, train_labels))

# Zamiana label na factor
train$label <- as.factor(train$label)

# Budowa modelu + predykcja
randomForestModel <- randomForest(label ~ ., train[,c("V106", "V129", "V154", "V242", "V282", "V29", "V319", "V337", "V339", "V379", "V434", "V443", "V452", "V473", "V476", "V49", "label")], ntree = 100)
randomForestProb <- predict(randomForestModel, newdata = valid, type = 'prob')[,2]

# Zapis wynikow do pliku
results <- c("\"AGAPAL\"", randomForestProb)
write(results, file = "AGAPAL_artificial_prediction.txt")


#Import Libraries
library(RevoUtilsMath) #Parallel Computing Package
library(dplyr) # Data manipulation

library(ggthemes) #Plots
library(ggplot2)

library(mice) #Filling in missing data

library(caret) #my preferred modeling package for Regression & Classification

#Import Data
setwd("C:/Users/perus/OneDrive/GitHub/kaggle-solutions/Titanic") #For Laptop
train <- read.csv("train.csv")
test <- read.csv("test.csv")

#Factorize discrete predictors
trainFeatures <- c('Survived', 'Bin_Sex', 'Pclass')


train <- train %>%
  mutate(Bin_Sex = as.integer(Sex == "female"))
test <- test %>%
  mutate(Bin_Sex = as.integer(Sex == "female"))
  
trainSet <- dplyr::select(train, Survived, Bin_Sex, Pclass, Fare)
trainSet[trainFeatures] <- lapply(trainSet[trainFeatures], function(x) as.factor(x))

#Train/Test Split
inTraining <- createDataPartition(trainSet$Survived, times = 1, p = .75, list = FALSE)

trainModel <- trainSet[inTraining,]
testModel <- trainSet[-inTraining,]

#Set the validation method
set.seed(1423)
cvControl <- trainControl(method = "repeatedcv",
                          number = 669/2,
                          repeats = 20)

#train Model
glmFit1 <- train(Survived ~ Bin_Sex + Pclass + Fare, data = trainModel,
                 method = "bayesglm",
                 trControl = cvControl)

glmFit1

#Evaluate 
testPred <- predict(glmFit1, testModel)

postResample(testPred, testModel$Survived)

confusionMatrix(testPred, testModel$Survived)

#Produce Final Guess
test <- test %>%
  mutate(Bin_Sex = as.integer(Sex == "female"))
testFeatures <- c('Bin_Sex', 'Pclass')
testSet <- select(test, Bin_Sex, Pclass, Fare)
testSet[testFeatures] <- lapply(testSet[testFeatures], function(x) as.factor(x))

prediction <- predict(glmFit1, testSet, na.action = na.pass)

solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

write.csv(solution, file = 'glm_Solution.csv', row.names = FALSE)

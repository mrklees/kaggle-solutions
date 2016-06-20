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
train <- train %>%
  mutate(Bin_Sex = as.integer(Sex == "female"))

usefulFeatures <- c('Survived', 'Bin_Sex', 'Pclass', 'Age', 'SibSp','Parch', 'Embarked')
factorFeatures <- c('Bin_Sex', 'Pclass', 'Embarked', 'Survived')
train[factorFeatures] <- lapply(train[factorFeatures], function(x) as.integer(x)) #we're converting to integers for preprocessing

x <- select(train, Survived, Bin_Sex, Pclass, Age, SibSp, Parch, Embarked)
y <- select(train, Survived)

normalization <- preProcess(x, method = c('knnImpute'))
x <- predict(normalization, x)
x[factorFeatures] <- lapply(x[factorFeatures], function(x) as.factor(x))
x <- as.data.frame(x)

subsets <- c(1:6)

set.seed(101)

ctrl <- rfeControl(functions = lmFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x, y,
                 sizes = subsets,
                 rfeControl = ctrl)


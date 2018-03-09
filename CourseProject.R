library(c( "doSNOW", "ipred","xgboost"))

library(e1071)
library(caret)
library(dplyr)
library(corrplot)
library(doSNOW)

setwd("C:/Users/apiccioni/Documents/MINE/PracticalML")
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
# or
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

#=================================================================
# Data exploration
#=================================================================

str(training)
summary(training)

# the dependent variable is "classe", factor with 5 levels: A,B,C,D,E.
head(training$classe)
class(training$classe)
sapply(training, class)

# NA values. How many per variable?

NAonly <- training %>%
  select(which(colSums(is.na(.))>0))



#==================================================================
# Data Transformation
#==================================================================
# Putting the 2 data sets together to explore correlation, missing values and skweness
# We remove the first 2 columns, which hold the serial number of the observation and 
# the name of the individual doing the exercise; and we cut the dependent variable off
train1 <- training[,-c(1,2,160)]
test1 <- testing [,-c(1,2,160)]
data1 <- rbind(train1, test1)


# We replace NA's with "0", to allow further analysis
data1 <- data1 %>%
  mutate_if(is.factor, as.numeric) %>%
  replace(., is.na(.), 0)

colSums(is.na(data1))

# Let's run cor() on all the data set
correlations <- cor(data1)
corrplot(correlations[1:25,1:25], order = "hclust")

highCorr <- findCorrelation(correlations, cutoff = 0.8)
length(highCorr)

# filtering data to remove some parameters
filteredData<- data1[,-highCorr]
correlations <- cor(filteredData)
corrplot(correlations[1:25,1:25], order = "hclust")

# is there any outlier or particularly skewed data?
skewValues <- apply(filteredData, 2, skewness)
barplot(skewValues)

# We could transform the data to reduce skewness, by applying the preProcess() in caret

transData <- preProcess(filteredData,
                         method = c("BoxCox", "pca"))


# We can now apply the transformations:
transformed <- predict(transData, filteredData)
head(transformed[,1:10])
length(transformed) # 52 parameters are the result of this transformation

# Rebuild training and testing data-sets:
transTrain <- transformed[1:19622,]
transTrain$classe <- training$classe

transTest <- transformed[19623:19642,]
transTest$problem_id <- testing$problem_id

# we could look at the outcome to see that the classes are balanced (no need to downsample?)
table(transTrain$classe)


#=================================================================================
# Downsampling
#=================================================================================
# If we think that the frequencies above are not balanced and we want to subsample
# we can set up caret to perform a 10-fold CV repeated 3 times, for example, 
# and use a grid search for optimal model hyperparameter values, like here:

train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = "grid")
# Leverage a grid search of hyperparameters for xgboost. See the following 

tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)
View(tune.grid)


# NOTE - Tune this number based on the number of cores/threads
# available on your machine!!!
# cl <- makeCluster(10, type = "SOCK")
cl <- makeCluster(2, type = "SOCK") # I set here "2", one for each cpu core of the laptop

# Register clusters so that caret will know to train in parallel.
registerDoSNOW(cl)

#=================================================================================
# Train the model with SVM
#=================================================================================
#  Remove dependent variable from data set to run svm(e1071)
y <- transTrain$classe
x <- transTrain[,-53]

svmFit <- svm(x, y, scale = FALSE,
              type = "C-classification",
              cross = 10,
              kernel = "radial")

preds <- predict(svmFit, transTest[,-53])
preds

# RESULT 
#19623 19624 19625 19626 19627 19628 19629 19630 19631 19632 19633 19634 19635 19636 19637 19638 19639 19640 
#B     A     C     A     A     E     D     B     A     A     B     C     B     A     E     E     A     B 
#19641 19642 
#B     B 

# 19 out of 20 were correct! The answer for 19625 is B and not C.

summary(svmFit)

#=================================================================================
# Train the model with xgboost
#=================================================================================



# Train the xgboost model unsing 10-fold CV repeated 3 times and
# a hyperparameter grid search to train the optimal model.
caret.cv <- train(classe ~ .,
                  data = transTrain,
                  method = "xgbTree",
                  tuneGrid = tune.grid,
                  trControl = train.control)
stopCluster(cl)

# Examine caret's 
caret.cv

# try to predict
preds2 <- predict(caret.cv, transTest)
confusionMatrix(preds, preds2, dnn = c("XGB", "SVM"))

library(tidyverse)
library(magrittr)
library(effectsize)
library(zeallot)

get_data_file_paths <- function(dataset){
  temp_DATA_DIRECTORY_PATH = file.path("/data/DropoutUncertaintyExps/UCI_Datasets",dataset,"data")
  temp_INDEX_FEATURES_FILE = file.path(temp_DATA_DIRECTORY_PATH,"index_features.txt")
  temp_INDEX_TARGET_FILE = file.path(temp_DATA_DIRECTORY_PATH,"index_target.txt")
  temp_DATA_FILE = file.path(temp_DATA_DIRECTORY_PATH,"data.txt")
  return(list(temp_DATA_DIRECTORY_PATH, temp_INDEX_FEATURES_FILE, temp_INDEX_TARGET_FILE, temp_DATA_FILE))
}


get_index_train_test_path <- function(DATA_DIRECTORY_PATH, split_num, train = T){
  if(train){
    return(file.path(DATA_DIRECTORY_PATH, paste0("index_train_",split_num ,".txt")))
  }
  else{
    return(file.path(DATA_DIRECTORY_PATH, paste0("index_test_",split_num,".txt")))
  }
}

get_data <- function(dataset){
  c(DATA_DIRECTORY_PATH, INDEX_FEATURES_FILE, INDEX_TARGET_FILE, DATA_FILE) %<-% get_data_file_paths(dataset)
  data <- tibble(read.table(DATA_FILE))
  index_features <- read.table(INDEX_FEATURES_FILE)
  index_target <- read.table(INDEX_TARGET_FILE)
  
  # +1 because R is 1-indexed
  X <- data[,index_features$V1+1]
  y <- data[,index_target$V1+1]
  colnames(y) <- c('Y')
  return(list(X=X,y=y))
}

get_data_splits <- function(dataset, split){
  c(X, y) %<-% get_data(dataset)
  c(DATA_DIRECTORY_PATH, INDEX_FEATURES_FILE, INDEX_TARGET_FILE, DATA_FILE) %<-% get_data_file_paths(dataset)
  index_train <- read_table(get_index_train_test_path(DATA_DIRECTORY_PATH, split, train=T), col_names = F, col_types = c('i'))
  index_test <- read_table(get_index_train_test_path(DATA_DIRECTORY_PATH, split, train=F), col_names = F, col_types = c('i'))
  
  X_trainval <- X[index_train$X1,]
  y_trainval <- y[index_train$X1,]
  
  X_test <- X[index_test$X1,]
  y_test <- y[index_test$X1,]
  
  num_training_examples <- round(0.8 * dim(X_trainval)[1])
  
  X_train <- X_trainval[seq(num_training_examples),]
  y_train <- y_trainval[seq(num_training_examples),]
  
  X_val <- X_trainval[-seq(num_training_examples),]
  y_val <- y_trainval[-seq(num_training_examples),]

  return(list(X_train=X_train,
              y_train=y_train,
              X_val=X_val,
              y_val=y_val,
              X_test=X_test,
              y_test=y_test))
}

for(DATASET in list.files('/data/DropoutUncertaintyExps/UCI_Datasets/')){
  for(SPLIT in 0:19){
    if(DATASET=='protein-tertiary-structure' && SPLIT>4){
      next
    }
    print(paste(DATASET, SPLIT))
    c(X_train, y_train, X_val, y_val, X_test, y_test) %<-% get_data_splits(DATASET, SPLIT)
    lm_Data <- bind_cols(bind_rows(X_train,X_val), bind_rows(y_train,y_val))
    model <- standardize(lm(Y~., lm_Data))
    
    train_preds_norm <- predict.lm(model, standardize(lm_Data %>% select(-Y)), interval='prediction')
    test_preds_norm <- predict.lm(model, standardize(X_test), interval='prediction')
    
    train_preds <- train_preds_norm*sd(lm_Data$Y)+mean(lm_Data$Y)
    test_preds <- test_preds_norm*sd(lm_Data$Y)+mean(lm_Data$Y)
    
    #calculate RMSE 
    train_rmse <- sqrt(mean((train_preds[,1]-lm_Data$Y)^2))
    test_rmse <- sqrt(mean((test_preds[,1]-y_test$Y)^2))
    
    #calculate width in terms of SDs 
    train_width <- mean(train_preds_norm[,3] - train_preds_norm[,2])
    test_width <- mean(test_preds_norm[,3] - test_preds_norm[,2])
    
    #calculate coverage 
    
    train_coverage <- mean(train_preds[,2]<lm_Data$Y & lm_Data$Y<train_preds[,3])
    test_coverage <- mean(test_preds[,2]<y_test$Y & y_test$Y<test_preds[,3])
    
    write_csv(tibble(dataset=DATASET, method='linear_regression', split=SPLIT, traintest='test', coverage=test_coverage, width=test_width, rmse=test_rmse), path='/data/processed/lr.csv', append=T)
    write_csv(tibble(dataset=DATASET, method='linear_regression', split=SPLIT, traintest='train', coverage=train_coverage, width=train_width, rmse=train_rmse),path='/data/processed/lr.csv', append=T)
  }
}








library(tidyverse)
library(magrittr)
library(stringr)

# mnist_data <- readr::read_delim('mnist_coverage_widths.csv', delim=',')
# cifar_data <- readr::read_delim('cifar_coverage_widths.csv', delim=',')

mnist_data <- readr::read_delim('/data/processed/mnist_coverage_widths.csv', delim=',')
cifar_data <- readr::read_delim('data/processed/cifar_coverage_widths.csv', delim=',')

create_tidy_df <- function(dataset, split){
  
  dataset_name <- deparse(substitute(dataset))
  
  if(split=='standard'){
    standard_df <- dataset %>% filter(split=='test' | split=='train' | split=='valid')
    standard_df$ordering <- NA
    return(standard_df)
  }
  
  else if(split=='fashion_mnist'){
    split_df <- dataset %>% filter(split=='fashion_mnist')
    split_df$ordering <- NA 
  }
  
  else if(split=='not_mnist'){
    split_df <- dataset %>% filter(split=='fashion_mnist')
    split_df$ordering <- NA 
  }
  
  else if(split=='svhn'){
    split_df <- dataset %>% filter(split=='svhn')
    split_df$ordering <- NA 
  }
  
  else if(split=='rot'){
    rows <- str_detect(dataset$split, 'rot_*')
    split_df <- dataset %>% filter(rows)
    ordering <- as.numeric(unlist(str_split(split_df$split, '_'))[seq(2, 2*length(split_df$split),2)])
    split_df$ordering <- ordering 
    

  }
  
  else if(split=='roll'){
    rows <- str_detect(dataset$split, 'roll_*')
    split_df <- dataset %>% filter(rows)
    ordering <- as.numeric(unlist(str_split(split_df$split, '_'))[seq(2, 2*length(split_df$split),2)])
    split_df$ordering <- ordering 
    

  }
  else if(split=='cifar_roll'){
    rows <- str_detect(dataset$split, 'roll-*')
    split_df <- dataset %>% filter(rows)
    ordering <- as.numeric(unlist(str_split(split_df$split, '-'))[seq(2, 2*length(split_df$split),2)])
    split_df$ordering <- ordering 
    
  }
  
  # otherwise, it's one of 16 transformations for CIFAR 
  else{
    rows <- str_detect(dataset$split, split)
    split_df <- dataset %>% filter(rows)
    ordering <- as.numeric(unlist(str_split(split_df$split, '-'))[seq(4, 4*length(split_df$split),4)])
    split_df$ordering <- ordering 
    
  }
  
  return(split_df)
}

mnist_splits <- c('standard', 'fashion_mnist', 'not_mnist', 'roll', 'rot')
mnist_tidy <- tibble()
for(split in mnist_splits){
  mnist_tidy <- rbind(mnist_tidy, create_tidy_df(mnist_data, split))
}

cifar_splits <- c('standard', 'svhn', 'cifar_roll', 'brightness', 
                  'contrast', 'defocus_blur', 'elastic_transform', 
                  'fog', 'frost', 'gaussian_blur', 
                  'gaussian_noise', 'glass_blur', 'impulse_noise', 
                  'pixelate', 'saturate', 'shot_noise', 
                  'spatter', 'speckle_noise', 'zoom_blur')
cifar_tidy <- tibble()
for(split in cifar_splits){
  cifar_tidy <- rbind(cifar_tidy, create_tidy_df(cifar_data, split))
}
cifar_tidy$dataset <- 'cifar'
mnist_tidy$dataset <- 'mnist'

# in the CIFAR data there was an error in the implementation of dropout
cifar_tidy %<>% filter(method!='dropout')

uq_tidy <- rbind(mnist_tidy, cifar_tidy)

uq_tidy$method[uq_tidy$method=='dropout_nofirst'] <- 'dropout'


saveRDS(object = uq_tidy, file='data/processed/uq_tidy.rds')

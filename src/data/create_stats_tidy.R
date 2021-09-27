library(tidyverse)
library(magrittr)

mnist_brier <- read_csv('./data/processed/mnist_brier.csv') %>%
pivot_longer(., X1, values_to = 'split') %>% 
  select(-name) %>%
  pivot_longer(data = ., 
               cols = -split, 
               values_to = 'brier', 
               names_to = 'method')

cifar_brier <- read_csv('./data/processed/cifar_brier.csv') %>%
  pivot_longer(., X1, values_to = 'split') %>% 
  select(-name) %>%
  pivot_longer(data = ., 
               cols = -split, 
               values_to = 'brier', 
               names_to = 'method')

mnist_ece <- read_csv('./data/processed/mnist_ece.csv') %>%
  pivot_longer(., X1, values_to = 'split') %>% 
  select(-name) %>%
  pivot_longer(data = ., 
               cols = -split, 
               values_to = 'ece', 
               names_to = 'method')

cifar_ece <- read_csv('./data/processed/cifar_ece.csv') %>%
  pivot_longer(., X1, values_to = 'split') %>% 
  select(-name) %>%
  pivot_longer(data = ., 
               cols = -split, 
               values_to = 'ece', 
               names_to = 'method')

mnist_brier %<>% inner_join(mnist_ece)

mnist_splits <- c('standard', 'fashion_mnist', 'not_mnist', 'roll', 'rot')
mnist_brier_tidy <- tibble()
for(split in mnist_splits){
  mnist_brier_tidy <- rbind(mnist_brier_tidy, create_tidy_df(mnist_brier %>% inner_join(mnist_ece), split))
}
mnist_brier_tidy$dataset <- 'mnist'

cifar_brier %<>% inner_join(cifar_ece)
cifar_splits <- c('standard', 'svhn', 'cifar_roll', 'brightness', 
                  'contrast', 'defocus_blur', 'elastic_transform', 
                  'fog', 'frost', 'gaussian_blur', 
                  'gaussian_noise', 'glass_blur', 'impulse_noise', 
                  'pixelate', 'saturate', 'shot_noise', 
                  'spatter', 'speckle_noise', 'zoom_blur')
cifar_tidy <- tibble()
for(split in cifar_splits){
  cifar_tidy <- rbind(cifar_tidy, create_tidy_df(cifar_brier, split))
}
cifar_tidy$dataset <- 'cifar'
cifar_tidy %<>% filter(method!='dropout')

stats_tidy <- rbind(mnist_brier_tidy, cifar_tidy)

stats_tidy$method[stats_tidy$method=='dropout_nofirst'] <- 'dropout'
saveRDS(object = stats_tidy, file='./data/processed/stats_tidy.rds')

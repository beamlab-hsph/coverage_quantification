library(tidyverse)
library(magrittr)
library(stringr)


imagenet_data <- readr::read_delim('./data/processed/imagenet_coverage_widths.csv', delim=',')

create_tidy_df <- function(dataset, split){
  
  dataset_name <- deparse(substitute(dataset))
  
  if(split=='standard'){
    standard_df <- dataset %>% filter(split=='test' | split=='train' | split=='valid')
    standard_df$ordering <- NA
    return(standard_df)
  }
  
  else if(split=='celeb_a'){
    split_df <- dataset %>% filter(split=='celeb_a')
    split_df$ordering <- NA 
  }
  
  # otherwise, it's one of 16 transformations for Imagemnet  
  else{
    rows <- str_detect(dataset$split, split)
    split_df <- dataset %>% filter(rows)
    ordering <- as.numeric(unlist(str_split(split_df$split, '-'))[seq(4, 4*length(split_df$split),4)])
    split_df$ordering <- ordering 
    
  }
  
  return(split_df)
}

imagenet_splits <- c('standard', 'celeb_a', 'brightness', 
                  'contrast', 'defocus_blur', 'elastic_transform', 
                  'fog', 'frost', 'gaussian_blur', 
                  'gaussian_noise', 'glass_blur', 'impulse_noise', 
                  'pixelate', 'saturate', 'shot_noise', 
                  'spatter', 'speckle_noise', 'zoom_blur')
imagenet_tidy <- tibble()
for(split in imagenet_splits){
  imagenet_tidy <- rbind(imagenet_tidy, create_tidy_df(imagenet_data, split))
}
imagenet_tidy$dataset <- 'imagenet'

saveRDS(object = imagenet_tidy, file='./data/processed/R/coverage_properties/imagenet_tidy.rds')

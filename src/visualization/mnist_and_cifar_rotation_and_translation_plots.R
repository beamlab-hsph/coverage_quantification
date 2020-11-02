library(tidyverse)
library(magrittr)
library(stringr)
library(cowplot)
library(kableExtra)
library(egg)

uq_tidy <- readRDS('/data/processed/uq_tidy.rds')


create_coverage_plot <- function(dataset_name, split, eps_level){
  selected_df <- uq_tidy %>% filter(dataset==dataset_name & eps==eps_level)
  
  #want to include this on all plots 
  standard_df <- selected_df %>% filter(split=='test' | split=='train' | split=='valid')
  
  ordered_plot <- FALSE
  
  split_text <- ''
  
  if(split=='fashion_mnist'){
    split_df <- selected_df %>% filter(split=='fashion_mnist')
    filtered_df <- split_df
    split_text <- 'Fashion MNIST'
  }
  
  else if(split=='not_mnist'){
    split_df <- selected_df %>% filter(split=='fashion_mnist')
    filtered_df <- split_df
    split_text <- 'Not MNIST'
  }
  
  else if(split=='svhn'){
    split_df <- selected_df %>% filter(split=='svhn')
    filtered_df <- split_df
    split_text <- 'SVHN'
  }
  
  else if(split=='rot'){
    rows <- str_detect(selected_df$split, 'rot_*')
    split_df <- selected_df %>% filter(rows)
    
    #test is 0 degrees rotation 
    standard_df$ordering <- 0 
    standard_df$ordering[standard_df$split=='valid'] <- -15
    standard_df$ordering[standard_df$split=='train'] <- -30
    
    ordered_plot <- TRUE 
    
    filtered_df <- split_df
    split_text <- 'Rotation Shift'
  }
  
  else if(split=='roll'){
    rows <- str_detect(selected_df$split, 'roll_*')
    split_df <- selected_df %>% filter(rows)
    
    
    #test is 0 degrees rolling 
    standard_df$ordering <- 0 
    standard_df$ordering[standard_df$split=='valid'] <- -2
    standard_df$ordering[standard_df$split=='train'] <- -4
    
    ordered_plot <- TRUE 
    
    filtered_df <- split_df
    split_text <- 'Rolling Shift'
  }
  else if(split=='cifar_roll'){
    rows <- str_detect(selected_df$split, 'roll-*')
    split_df <- selected_df %>% filter(rows)
    
    #test is 0 degrees rolling 
    standard_df$ordering <- 0 
    standard_df$ordering[standard_df$split=='valid'] <- -4
    standard_df$ordering[standard_df$split=='train'] <- -8
    
    ordered_plot <- TRUE 
    
    filtered_df <- split_df
    split_text <- 'Rolling Shift'
  }
  
  # otherwise, it's one of 16 transformations for CIFAR 
  else{
    rows <- str_detect(selected_df$split, split)
    split_df <- selected_df %>% filter(rows)
    
    #test is 0 shift 
    standard_df$ordering <- 0 
    standard_df$ordering[standard_df$split=='valid'] <- -1
    standard_df$ordering[standard_df$split=='train'] <- -2
    
    ordered_plot <- TRUE 
    
    filtered_df <- split_df
    split_text <- split
  }
  
  size_factor <- 5
  if(ordered_plot){
    p <- ggplot(filtered_df, aes(x=ordering, y=coverage, color=factor(method)))+scale_x_continuous(breaks = round(seq(min(filtered_df$ordering), max(filtered_df$ordering), length.out=length(unique(filtered_df$ordering))),1))
    if(split=='rot'){
      p <- p + scale_x_continuous(name='Shift', breaks = seq(15, 180, 15), labels = c(sapply(seq(15, 180, 15), function(x) paste0(x, '°')) ))
    }
    if(split=='roll'){
      p <- p + scale_x_continuous(name='Shift', breaks = seq(2, 28, 2), labels = c( sapply(seq(2, 28, 2), function(x) paste0(x, 'px')) ))
    }
    if(split=='cifar_roll'){
      p <- p + scale_x_continuous(name='Shift', breaks = seq(4, 28, 4), labels = c(sapply(seq(4, 28, 4), function(x) paste0(x, 'px')) ))
    }
  }
  else{
    p <- ggplot(filtered_df, aes(x=reorder(split,coverage,max), y=coverage, color=factor(method)))+theme(axis.text.x = element_text(angle = 90))
  }
  
  p <- p + 
    geom_point(alpha=.5)+
    theme_bw()+
    xlab('Shift')+
    ylab('Coverage')+
    labs(color='Method')+
    geom_hline(yintercept = 1-eps_level, color='black')+
    ylim(c(0, 1))+
    scale_color_discrete(name="Method",
                         breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                         labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))
  
  
  return(list(plot=p, filtered_df=filtered_df, standard_df=standard_df))
}

scaleFUN <- function(x) sprintf("%.2f", x)

create_width_plot <- function(dataset_name, split, eps_level){
  
  selected_df <- uq_tidy %>% filter(dataset==dataset_name & eps==eps_level)
  
  #want to include this on all plots 
  standard_df <- selected_df %>% filter(split=='test' | split=='train' | split=='valid')
  
  ordered_plot <- FALSE
  
  split_text <- ''
  
  if(split=='fashion_mnist'){
    split_df <- selected_df %>% filter(split=='fashion_mnist')
    filtered_df <- split_df
    split_text <- 'Fashion MNIST'
  }
  
  else if(split=='not_mnist'){
    split_df <- selected_df %>% filter(split=='fashion_mnist')
    filtered_df <- split_df
    split_text <- 'Not MNIST'
  }
  
  else if(split=='svhn'){
    split_df <- selected_df %>% filter(split=='svhn')
    filtered_df <- split_df
    split_text <- 'SVHN'
  }
  
  else if(split=='rot'){
    rows <- str_detect(selected_df$split, 'rot_*')
    split_df <- selected_df %>% filter(rows)
    
    #test is 0 degrees rotation 
    standard_df$ordering <- 0 
    standard_df$ordering[standard_df$split=='valid'] <- -15
    standard_df$ordering[standard_df$split=='train'] <- -30
    
    ordered_plot <- TRUE 
    
    filtered_df <- split_df
    split_text <- 'Rotation Shift'
  }
  
  else if(split=='roll'){
    rows <- str_detect(selected_df$split, 'roll_*')
    split_df <- selected_df %>% filter(rows)
    
    #test is 0 degrees rolling 
    standard_df$ordering <- 0 
    standard_df$ordering[standard_df$split=='valid'] <- -2
    standard_df$ordering[standard_df$split=='train'] <- -4
    
    ordered_plot <- TRUE 
    
    filtered_df <- split_df
    split_text <- 'Rolling Shift'
  }
  else if(split=='cifar_roll'){
    rows <- str_detect(selected_df$split, 'roll-*')
    split_df <- selected_df %>% filter(rows)
    
    #test is 0 degrees rolling 
    standard_df$ordering <- 0 
    standard_df$ordering[standard_df$split=='valid'] <- -4
    standard_df$ordering[standard_df$split=='train'] <- -8
    
    ordered_plot <- TRUE 
    
    filtered_df <- split_df
    split_text <- 'Rolling Shift'
  }
  
  # otherwise, it's one of 16 transformations for CIFAR 
  else{
    rows <- str_detect(selected_df$split, split)
    split_df <- selected_df %>% filter(rows)
    
    #test is 0 shift 
    standard_df$ordering <- 0 
    standard_df$ordering[standard_df$split=='valid'] <- -1
    standard_df$ordering[standard_df$split=='train'] <- -2
    
    ordered_plot <- TRUE 
    
    filtered_df <- split_df
    split_text <- split
  }
  
  size_factor <- 5
  if(ordered_plot){
    p <- ggplot(filtered_df, aes(x=ordering, y=width, color=factor(method)))+scale_x_continuous(breaks = round(seq(min(filtered_df$ordering), max(filtered_df$ordering), length.out=length(unique(filtered_df$ordering))),1))
    if(split=='rot'){
      p <- p + scale_x_continuous(name='Shift', breaks = seq(15, 180, 15), labels = c( sapply(seq(15, 180, 15), function(x) paste0(x, '°')) ))
    }
    if(split=='roll'){
      p <- p + scale_x_continuous(name='Shift', breaks = seq(2, 28, 2), labels = c(sapply(seq(2, 28, 2), function(x) paste0(x, 'px')) ))
    }
    if(split=='cifar_roll'){
      p <- p + scale_x_continuous(name='Shift', breaks = seq(4, 28, 4), labels = c(sapply(seq(4, 28, 4), function(x) paste0(x, 'px')) ))
    }
  }
  else{
    p <- ggplot(filtered_df, aes(x=reorder(split,coverage,max), y=width, color=factor(method)))+theme(axis.text.x = element_text(angle = 90))
  }
  
  
  p <- p + 
    geom_point(alpha=.5)+
    theme_bw()+
    xlab('Shift')+
    ylab('Width')+
    labs(color='Method')+ scale_y_continuous(labels=scaleFUN)+
    ylim(1, ifelse(dataset_name=='mnist', 8, 3))+
    scale_color_discrete(name="Method",
                         breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                         labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))
  
  
  
  return(list(plot=p, filtered_df=filtered_df, standard_df=standard_df))
}

EPS = .05



mnist_rot_coverage <- create_coverage_plot('mnist', 'rot', .05)['plot']
ggsave('/reports/figures/tidy_data/mnist_rot_coverage_no_standard.png', width=10, height=4, units='in', dpi=300)



mnist_rot_width <- create_width_plot('mnist', 'rot', .05)['plot']
ggsave('/reports/figures/mnist_rot_width_no_standard.png', width=10, height=4, units='in', dpi=300)

create_entropy_plot('mnist', 'rot', .05)['plot']
ggsave('/reports/figures/mnist_rot_entropy_no_standard.png', width=10, height=4, units='in', dpi=300)




tiff('/reports/figures/mnist_roll_coverage.tiff', width=10, height=4, units='in', res=300)
mnist_roll_coverage <- create_coverage_plot('mnist', 'roll', .05)['plot']
dev.off()

tiff('/reports/figures/mnist_roll_width.tiff', width=10, height=4, units='in', res=300)
mnist_roll_width <- create_width_plot('mnist', 'roll', .05)['plot']
dev.off()


Axis_Theme <- theme(
  axis.text.x = element_text(size = 14, angle = 315),
  axis.text.y = element_text(size = 14),
  axis.title.y = element_text(size = 16) ,
  axis.title.x = element_text(size = 16) ,
  legend.text=element_text(size=12))

rot_y_theme <- theme(
  axis.title.y = element_blank())

fig1 <- ggarrange(mnist_rot_coverage$plot + theme(legend.position = "none") + Axis_Theme,
          mnist_roll_coverage$plot + 
            theme( axis.title.y = element_blank())+ Axis_Theme+rot_y_theme, 
          mnist_rot_width$plot + theme(legend.position = "none")+ Axis_Theme, 
          mnist_roll_width$plot+ 
            theme( axis.title.y = element_blank())+ Axis_Theme+rot_y_theme, 
          nrow=2)

ggsave('/reports/figures/mnist_fig1.png', plot=fig1, width=10, height=5.6, units='in', dpi=300)

fig1a <- ggarrange(mnist_rot_coverage$plot + theme(legend.position = "none") + Axis_Theme,
                  mnist_rot_width$plot + theme(legend.position = "none")+ Axis_Theme, 
                  nrow=2)

ggsave('/reports/figures/mnist_fig1a.png', plot=fig1a, width=10, height=12.26, units = 'in', dpi=300)

fig1b <- ggarrange(mnist_roll_coverage$plot + 
                     theme( axis.title.y = element_blank())+ Axis_Theme+rot_y_theme,
                   mnist_roll_width$plot+ 
                     theme( axis.title.y = element_blank())+ Axis_Theme+rot_y_theme, 
                   nrow=2)

ggsave('/reports/figures/mnist_fig1b.png', plot=fig1b, width=10, height=12.26, units = 'in', dpi=300)




cifar_coverage <- create_coverage_plot('cifar', 'cifar_roll', .05)['plot']

cifar_width <- create_width_plot('cifar', 'cifar_roll', .05)['plot']

fig2 <- ggarrange(cifar_coverage$plot + theme(legend.position = "none") + Axis_Theme,
                  cifar_width$plot+ 
                    theme( axis.title.y = element_blank())+ Axis_Theme+ theme(
                      axis.title.y = element_text(angle=90)), 
                  nrow=1)
ggsave('/reports/figures/cifar_fig2.png', plot=fig2, width=10, height=3.472, units = 'in', dpi=300)

# Table of iid MNIST 
mnist_standard <- uq_tidy %>% filter(dataset=='mnist' & eps==.05) %>% filter(split=='test' | split=='train' | split=='valid')

mnist_standard %<>% group_by(method, split) %>% summarise(mean_coverage = mean(coverage),
                                                         se_coverage = sd(coverage)/sqrt(n()), 
                                                         mean_width = mean(width), 
                                                         se_width = sd(width)/sqrt(n()))

train_df <- mnist_standard %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  unite(col='width',mean_width, se_width, sep=' ')%>% 
  filter(split=='train')

val_df <- mnist_standard %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  unite(col='width',mean_width, se_width, sep=' ')%>% 
  filter(split=='valid')

test_df <- mnist_standard %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  unite(col='width',mean_width, se_width, sep=' ')%>% 
  filter(split=='test')

bind_cols(train_df, val_df %>% select(-method), test_df %>% select(-method)) %>% 
  write_csv(., '/reports/tables/mnist_table.csv')

# Table of iid cifar 
cifar_standard <- uq_tidy %>% filter(dataset=='cifar' & eps==.05) %>% filter(split=='test' | split=='train' | split=='valid')

cifar_standard %<>% group_by(method, split) %>% summarise(mean_coverage = mean(coverage),
                                                          se_coverage = sd(coverage)/sqrt(n()), 
                                                          mean_width = mean(width), 
                                                          se_width = sd(width)/sqrt(n()))

train_df <- cifar_standard %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  unite(col='width',mean_width, se_width, sep=' ')%>% 
  filter(split=='train')

val_df <- cifar_standard %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  unite(col='width',mean_width, se_width, sep=' ')%>% 
  filter(split=='valid')

test_df <- cifar_standard %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  unite(col='width',mean_width, se_width, sep=' ')%>% 
  filter(split=='test')

bind_cols(train_df, val_df %>% select(-method), test_df %>% select(-method)) %>% 
  write_csv(., '/reports/tables/cifar_table.csv')

mnist_rot_df <- create_coverage_plot('mnist', 'rot', .05)[['filtered_df']]

# standard df is already filtered out
mnist_rot_df %>% 
  group_by(method) %>% 
  summarise(mean_coverage = mean(coverage),
            se_coverage = sd(coverage)/sqrt(n()), 
            mean_width = mean(width), 
            se_width = sd(width)/sqrt(n())) %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  unite(col='width',mean_width, se_width, sep=' ')%>% 
  write_csv(., '/reports/tables/mnist_rot.csv')

mnist_roll_df <- create_coverage_plot('mnist', 'roll', .05)[['filtered_df']]

# standard df is already filtered out
mnist_roll_df %>% 
  group_by(method) %>% 
  summarise(mean_coverage = mean(coverage),
            se_coverage = sd(coverage)/sqrt(n()), 
            mean_width = mean(width), 
            se_width = sd(width)/sqrt(n())) %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  unite(col='width',mean_width, se_width, sep=' ')%>% 
  write_csv(., '/reports/tables/mnist_roll.csv')

cifar_roll_df <- create_coverage_plot('cifar', 'cifar_roll', .05)[['filtered_df']]

# standard df is already filtered out
cifar_roll_df %>% 
  group_by(method) %>% 
  summarise(mean_coverage = mean(coverage),
            se_coverage = sd(coverage)/sqrt(n()), 
            mean_width = mean(width), 
            se_width = sd(width)/sqrt(n())) %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  unite(col='width',mean_width, se_width, sep=' ')%>% 
  write_csv(., '/reports/tables/cifar_roll.csv')
  
  

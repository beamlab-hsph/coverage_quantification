library(tidyverse)
library(magrittr)
library(stringr)
library(cowplot)
library(kableExtra)
library(plotrix)

eps_level <- .05

uq_tidy <- readRDS('~/R/coverage_properties/tidy_data/uq_tidy.rds')
cifar_df <- uq_tidy %>% filter(dataset=='cifar' & eps==eps_level)

uq_entropy <- readRDS('~/R/coverage_properties/uq_tidy_entropy.rds')
cifar_entropy_df <- uq_entropy %>% filter(dataset=='cifar')

# filter out rolling 
rows <- str_detect(cifar_df$split, 'roll-*')
cifar_df %<>% filter(!rows)
rows <- str_detect(cifar_entropy_df$split, 'roll-*')
cifar_entropy_df %<>% filter(!rows)


#filter out OOD 
rows <- str_detect(cifar_df$split, 'svhn')
cifar_df %<>% filter(!rows)
rows <- str_detect(cifar_entropy_df$split, 'svhn')
cifar_entropy_df %<>% filter(!rows)

#filter out train and val 
rows <- str_detect(cifar_df$split, 'train')
cifar_df %<>% filter(!rows)
rows <- str_detect(cifar_entropy_df$split, 'train')
cifar_entropy_df %<>% filter(!rows)

rows <- str_detect(cifar_df$split, 'valid')
cifar_df %<>% filter(!rows)
rows <- str_detect(cifar_entropy_df$split, 'valid')
cifar_entropy_df %<>% filter(!rows)

# add ordering for test
cifar_df$ordering[cifar_df$split=='test'] <- 0
cifar_entropy_df$ordering[cifar_entropy_df$split=='test'] <- 0

split_names <- c(
  `0` = "Test",
  `1` = "1",
  `2` = "2",
  `3` = "3",
  `4` = "4", 
  `5` = "5"
)

p_coverage <- ggplot(cifar_df, 
                     aes(x=factor(method), 
                         y=coverage, 
                         color=factor(method)))+
  ylab('Coverage')+
  geom_boxplot()+
  facet_grid(. ~ ordering, switch='x', labeller = as_labeller(split_names))+
  theme_bw()+
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())+
  xlab('Shift Intestity')+
  geom_hline(yintercept = 1-eps_level, color='black')+
  labs(color='Method')+
  ylim(0,1)+
  scale_color_discrete(name="Method",
                        breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                        labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))


p_coverage
ggsave('~/R/coverage_properties/tidy_data/cifar_corruption_coverage.png', device = 'png', p_coverage, width=10, height=4, units='in')

p_width <- ggplot(cifar_df, 
                     aes(x=factor(method), 
                         y=width, 
                         color=factor(method)))+
  ylab('Width')+
  geom_boxplot()+
  facet_grid(. ~ ordering, switch='x', labeller = as_labeller(split_names))+
  theme_bw()+
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())+
  xlab('Shift Intestity')+
  labs(color='Method')+
  scale_color_discrete(name="Method",
                       breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))

p_width
ggsave('~/R/coverage_properties/tidy_data/cifar_corruption_width.png', device = 'png', p_width, width=10, height=4, units='in')

p_entropy <- ggplot(cifar_entropy_df, 
                  aes(x=factor(method), 
                      y=entropy, 
                      color=factor(method)))+
  ylab('Entropy (Nats)')+
  ylim(0,2.5)+
  geom_boxplot()+
  facet_grid(. ~ ordering, switch='x', labeller = as_labeller(split_names))+
  theme_bw()+
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())+
  xlab('Shift Intestity')+
  labs(color='Method')+
  scale_color_discrete(name="Method",
                       breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))

p_entropy
ggsave('~/R/coverage_properties/tidy_data/cifar_corruption_entropy.png', device = 'png', p_entropy, width=10, height=4, units='in')



table_2_test <- cifar_df %>% filter(ordering==0) %>% group_by(method) %>% summarise(mean_coverage = mean(coverage), se_coverage = std.error(coverage), mean_widdth=mean(width), se_width=std.error(width) ) 
table_2 <- cifar_df %>% filter(ordering!=0) %>% group_by(method) %>% summarise(mean_coverage = mean(coverage), se_coverage = std.error(coverage), mean_widdth=mean(width), se_width=std.error(width) ) 

write_csv(cbind(table_2_test, table_2), '~/Downloads/table2.csv')

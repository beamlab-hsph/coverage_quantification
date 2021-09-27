library(tidyverse)
library(magrittr)
library(stringr)
library(cowplot)
library(kableExtra)
library(plotrix)

eps_level <- .05

imagenet_tidy <- readRDS('./data/processed/imagenet_tidy.rds')
imagenet_df <- imagenet_tidy %>% filter(dataset=='imagenet' & eps==eps_level)


#filter out OOD 
rows <- str_detect(imagenet_df$split, 'celeb_a')
imagenet_df %<>% filter(!rows)

#filter out train and val 
rows <- str_detect(imagenet_df$split, 'train')
imagenet_df %<>% filter(!rows)

rows <- str_detect(imagenet_df$split, 'valid')
imagenet_df %<>% filter(!rows)

# add ordering for test
imagenet_df$ordering[imagenet_df$split=='test'] <- 0

split_names <- c(
  `0` = "Test",
  `1` = "1",
  `2` = "2",
  `3` = "3",
  `4` = "4", 
  `5` = "5"
)

p_coverage <- ggplot(imagenet_df, 
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
ggsave('./reports/figures/imagenet_corruption_coverage.png', device = 'png', p_coverage, width=10, height=4, units='in')

p_width <- ggplot(imagenet_df, 
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
ggsave('./reports/figures/imagenet_corruption_width.png', device = 'png', p_width, width=10, height=4, units='in')

# p_entropy <- ggplot(imagenet_entropy_df, 
#                     aes(x=factor(method), 
#                         y=entropy, 
#                         color=factor(method)))+
#   ylab('Entropy (Nats)')+
#   ylim(0,2.5)+
#   geom_boxplot()+
#   facet_grid(. ~ ordering, switch='x', labeller = as_labeller(split_names))+
#   theme_bw()+
#   theme(axis.text.x=element_blank(),
#         axis.ticks.x=element_blank())+
#   xlab('Shift Intestity')+
#   labs(color='Method')+
#   scale_color_discrete(name="Method",
#                        breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
#                        labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))
# 
# p_entropy
# ggsave('~/R/coverage_properties/tidy_data/imagenet_corruption_entropy.png', device = 'png', p_entropy, width=10, height=4, units='in')



table_3_test <- imagenet_df %>% filter(ordering==0) %>% group_by(method) %>% summarise(mean_coverage = mean(coverage), se_coverage = std.error(coverage), mean_width=mean(width), se_width=std.error(width) ) 
table_3 <- imagenet_df %>% filter(ordering!=0) %>% group_by(method) %>% summarise(mean_coverage = mean(coverage), se_coverage = std.error(coverage), mean_width=mean(width), se_width=std.error(width) ) 

table_3 %>% 
  select(method, mean_coverage, se_coverage, mean_width, se_width) %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='width',mean_width, se_width, sep=' ') %>% 
  #pivot_wider(names_from=dataset, values_from = coverage) %>%
  write_csv(., '/reports/tables/imagenet_corruption_tables.csv')

table_3_test %>% 
  select(method, mean_coverage, mean_width) %>% 
  mutate(coverage = round(mean_coverage,4)) %>%
  mutate(width = round(mean_width,4)) %>%
  #pivot_wider(names_from=dataset, values_from = coverage) %>%
  write_csv(., './reports/tables/imagenet_corruption_testset.csv')


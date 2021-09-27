library(tidyverse)
library(magrittr)
library(stringr)
library(cowplot)
library(kableExtra)
library(egg)

base_dir <- '.' #currect wd should be the parent dir of the repo
uq_tidy <- readRDS(paste0(base_dir, '/data/processed/uq_tidy.rds'))
stats_tidy <- readRDS(paste0(base_dir, '/data/processed/stats_tidy.rds'))

uq_tidy %<>% select(-eps) %>% filter(dataset=='cifar')
stats_tidy %<>% filter(dataset=='cifar')

cifar_tidy <- left_join(stats_tidy, uq_tidy) 

# filter out rolling 
rows <- str_detect(cifar_tidy$split, 'roll-*')
cifar_tidy %<>% filter(!rows)

#filter out OOD 
rows <- str_detect(cifar_tidy$split, 'svhn')
cifar_tidy %<>% filter(!rows)

#filter out train and val 
rows <- str_detect(cifar_tidy$split, 'train')
cifar_tidy %<>% filter(!rows)

rows <- str_detect(cifar_tidy$split, 'valid')
cifar_tidy %<>% filter(!rows)

# add ordering for test
cifar_tidy$ordering[cifar_tidy$split=='test'] <- 0

split_names <- c(
  `0` = "Test",
  `1` = "1",
  `2` = "2",
  `3` = "3",
  `4` = "4", 
  `5` = "5"
)


cifar_tidy %<>% group_by(ordering, method) %>%
  summarise(mean_brier=mean(brier), 
            mean_ece=mean(ece), 
            mean_coverage=mean(coverage)) %>% ungroup

rankings_df <- cifar_tidy %>%
       group_by(ordering) %>% 
       arrange(ordering, mean_ece, method) %>%
       mutate(ece_rank = row_number()) %>%
       arrange(ordering, mean_brier, method) %>%
       mutate(brier_rank = row_number()) %>%
       arrange(ordering, desc(mean_coverage), method) %>%
       mutate(coverage_rank = row_number()) %>%
       ungroup() 

bs_plot <- ggplot(rankings_df %>% filter(ordering!=0), 
       aes(x=ordering, 
           y=brier_rank, 
           color=factor(method)))+
  geom_point()+
  geom_line()+
  scale_y_reverse(breaks = 7:1)+
  theme_bw()+
  ylab('Brier Score Rank')+
  xlab('Corruption Level')+
  scale_color_discrete(name="Method",
                       breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))

ece_plot <- ggplot(rankings_df %>% filter(ordering!=0), 
                   aes(x=ordering, 
                       y=ece_rank, 
                       color=factor(method)))+
  geom_point()+
  geom_line()+
  theme_bw()+
  ylab('ECE Rank')+
  scale_y_reverse(breaks = 7:1)+
  xlab('Corruption Level')+
  scale_color_discrete(name="Method",
                       breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))

coverage_plot <- ggplot(rankings_df %>% filter(ordering!=0), 
                   aes(x=ordering, 
                       y=coverage_rank, 
                       color=factor(method)))+
  geom_point()+
  geom_line()+
  theme_bw()+
  ylab('Coverage Rank')+
  scale_y_reverse(breaks = 7:1)+
  xlab('Corruption Level')+
  scale_color_discrete(name="Method",
                       breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))

combined_plot <- ggarrange(bs_plot+ theme(legend.position = "none"), ece_plot+ theme(legend.position = "none"), coverage_plot, nrow=1)
ggsave(paste0(base_dir, '/reports/figures/rank.png'), combined_plot, width=7.5, height=3)

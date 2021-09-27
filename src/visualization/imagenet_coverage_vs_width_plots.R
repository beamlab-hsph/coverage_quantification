library(tidyverse)

 
eps_level <- .05

base_dir <- '~/Repos/coverage-quantification/' #wd should be parent dir of the repo
imagenet_tidy <- readRDS(paste0(base_dir, '/data/processed/imagenet_tidy.rds'))
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

uq_tidy <- readRDS(paste0(base_dir, '/data/processed/uq_tidy.rds'))
cifar_df <- uq_tidy %>% filter(dataset=='cifar' & eps==eps_level)


# filter out rolling 
rows <- str_detect(cifar_df$split, 'roll-*')
cifar_df %<>% filter(!rows)


#filter out OOD 
rows <- str_detect(cifar_df$split, 'svhn')
cifar_df %<>% filter(!rows)

#filter out train and val 
rows <- str_detect(cifar_df$split, 'train')
cifar_df %<>% filter(!rows)


rows <- str_detect(cifar_df$split, 'valid')
cifar_df %<>% filter(!rows)

# add ordering for test
cifar_df$ordering[cifar_df$split=='test'] <- 0

imagenet_cifar <- rbind(imagenet_df, cifar_df)

cifar_wc_facet_plot <- ggplot(imagenet_cifar %>% filter(dataset=='cifar'), aes(y=coverage, x=width, color=method))+
  geom_point(alpha=.7, size =.4)+
  #geom_blank(data=imagenet_cifar%>% filter(dataset=='imagenet')) +
  facet_grid(. ~ ordering, switch='x', labeller = as_labeller(split_names))+
  scale_y_continuous(labels = scales::percent, limits=c(0,1))+
  theme_bw()+
  ylab('Coverage')+
  xlab('Width')+
  geom_hline(yintercept = 1-eps_level, color='black')+
  scale_color_discrete(name="Method",
                       breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))

cifar_wc_facet_plot
ggsave(paste0(base_dir,'/reports/figures/cifar_coverage_vs_width.png'), device = 'png', cifar_wc_facet_plot, width=10, height=4, units='in')


gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

colors_7 <- gg_color_hue(7)

fig_data <- imagenet_cifar %>% 
  filter(dataset=='cifar', ordering==5) %>%
  mutate(ensemble = method=='ensemble')

slope <--0.008134  
intercept <- 0.770986
cifar_wc_facet_5_plot <- ggplot(fig_data, aes(y=coverage, x=width, color=ensemble))+
  geom_point(alpha=.7, size =1)+
  #geom_blank(data=imagenet_cifar%>% filter(dataset=='imagenet')) +
  #facet_grid(. ~ ordering, switch='x', labeller = as_labeller(split_names))+
  scale_y_continuous(labels = scales::percent, limits = c(0,1))+
  theme_bw()+
  ylab('Coverage')+
  xlab('Width')+
  #geom_hline(yintercept = 1-eps_level, color='black')+
  scale_color_manual(name="Method",
  labels=c("Not Ensemble", "Ensemble"), 
  values=c("#999999", colors_7[2]))+
  geom_abline(slope= slope, intercept=intercept)

cifar_wc_facet_5_plot

ggsave(paste0(base_dir, '/reports/figures/revised_ensemble_5.png'), cifar_wc_facet_5_plot, height=4, width=7.5)

fig_data %>%
  mutate(prediction = slope*width+intercept) %>% 
  mutate(better = prediction<coverage) %>% 
  group_by(method) %>% 
  summarise(mean_better=mean(better))

cifar_data <- imagenet_cifar %>% 
  filter(dataset=='cifar')

imagenet_data <- imagenet_cifar %>% 
  filter(dataset=='imagenet')

cifar_lm <- cifar_data %>% 
  group_by(ordering) %>% 
  do(tidy(lm(coverage ~ width, .)))

imagenet_lm <- imagenet_data %>% 
  group_by(ordering) %>%
  do(tidy(lm(coverage ~ width, .)))


get_lr_percent <- function(df, lm_df, ordering_){
  ret_df <- df %>%
    filter(ordering==ordering_) %>%
    mutate(prediction = lm_df$estimate[ordering_*2+2]*width+lm_df$estimate[ordering_*2+1]) %>% 
    mutate(better = prediction<coverage) %>% 
    group_by(method) %>% 
    summarise(mean_better=mean(better))
  ret_df$ordering <- ordering_
  return(ret_df)
}

cifar_percents <- do.call("rbind", lapply(1:5, function(i){get_lr_percent(cifar_data, cifar_lm, i)}))

imagenet_percents <- do.call("rbind", lapply(1:5, function(i){get_lr_percent(imagenet_data, imagenet_lm, i)}))

cifar_percents_plot <- ggplot(cifar_percents, aes(x=ordering, y=mean_better, color=factor(method))) + 
  geom_point()+
  geom_line()+
  theme_bw()+
  ylab('Relative Performance')+
  xlab('Corruption Level')+
  scale_color_discrete(name="Method",
                       breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))

ggsave(paste0(base_dir, '/reports/figures/cifar_percents.png'), cifar_percents_plot, width=7.5, height=3)

imagenet_percent_plots <- ggplot(imagenet_percents, aes(x=ordering, y=mean_better, color=factor(method))) + 
  geom_point()+
  geom_line()+
  scale_color_manual(values =colors_7[-5], name="Method",
                     breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "temp_scaling", 'vanilla'),
                     labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "Temp Scaling", "Vanilla"))+
  theme_bw()+
  ylab('Relative Performance')+
  xlab('Corruption Level')

ggsave(paste0(base_dir, '/reports/figures/imagenet_percents.png'), imagenet_percent_plots, width=7.5, height=3)


imagenet_wc_facet_plot <- ggplot(imagenet_cifar %>% filter(dataset=='imagenet'), aes(y=coverage, x=width, color=method))+
  geom_point(alpha=.7, size =.4)+
  facet_grid(. ~ ordering, switch='x', labeller = as_labeller(split_names))+
  theme_bw()+
  ylab('Coverage')+
  xlab('Width')+
  scale_y_continuous(labels = scales::percent, limits=c(0,1))+
  geom_hline(yintercept = 1-eps_level, color='black')+
  scale_color_manual(values =colors_7[-5], name="Method",
                       breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "Temp Scaling", "Vanilla"))

imagenet_wc_facet_plot

ggsave(paste0(base_dir, '/reports/figures/imagenet_coverage_vs_width.png'), device = 'png', imagenet_wc_facet_plot, width=10, height=4, units='in')

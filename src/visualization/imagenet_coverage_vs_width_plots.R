library(tidyverse)

 
eps_level <- .05

imagenet_tidy <- readRDS('/data/processed/imagenet_tidy.rds')
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

uq_tidy <- readRDS('/data/processed/uq_tidy.rds')
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

imagenet_cifar <- rbind(imagenet_df, cifar_df)

cifar_wc_facet_plot <- ggplot(imagenet_cifar %>% filter(dataset=='cifar'), aes(y=coverage, x=width, color=method))+
  geom_point(alpha=.7, size =.4)+
  #geom_blank(data=imagenet_cifar%>% filter(dataset=='imagenet')) +
  facet_grid(. ~ ordering, switch='x', labeller = as_labeller(split_names))+
  ylim(c(0,1))+
  theme_bw()+
  ylab('Coverage')+
  xlab('Width')+
  geom_hline(yintercept = 1-eps_level, color='black')+
  scale_color_discrete(name="Method",
                       breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))

cifar_wc_facet_plot

ggsave('/reports/figures/cifar_coverage_vs_width.png', device = 'png', cifar_wc_facet_plot, width=10, height=4, units='in')


gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

colors_7 <- gg_color_hue(7)


imagenet_wc_facet_plot <- ggplot(imagenet_cifar %>% filter(dataset=='imagenet'), aes(y=coverage, x=width, color=method))+
  geom_point(alpha=.7, size =.4)+
  facet_grid(. ~ ordering, switch='x', labeller = as_labeller(split_names))+
  theme_bw()+
  ylab('Coverage')+
  xlab('Width')+
  ylim(c(0,1))+
  geom_hline(yintercept = 1-eps_level, color='black')+
  scale_color_manual(values =colors_7[-5], name="Method",
                       breaks=c("dropout", "ensemble", "ll_dropout", "ll_svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "LL Dropout", "LL SVI", "Temp Scaling", "Vanilla"))

imagenet_wc_facet_plot

ggsave('/reports/figures/imagenet_coverage_vs_width.png', device = 'png', imagenet_wc_facet_plot, width=10, height=4, units='in')

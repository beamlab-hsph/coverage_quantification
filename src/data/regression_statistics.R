library(tidyverse)
library(magrittr)

base_dir <- '~/Repos/coverage_quantification' # wd should the parent dir of the repo

lr_df <- read_csv(paste0(base_dir, '/data/processed/lr.csv'))
bayesian_df <- read_csv(paste0(base_dir, '/data/processed/model_output_summary_non_svi_eb.csv'), 
                        col_names = T)
svi_df <- read_csv(paste0(base_dir, '/data/processed/model_output_summary_svi_eb.csv'),
                   col_names = T)

gp_df <- read_csv(paste0(base_dir, '/data/processed/model_output_summary_gp.csv'), col_names = T)

combined_df <- bind_rows(lr_df, bayesian_df)

combined_df %<>% bind_rows(combined_df, svi_df, gp_df)

combined_df %<>% group_by(dataset, method, traintest) %>% 
  summarise(mean_coverage = mean(coverage),
            se_coverage = sd(coverage)/sqrt(n()), 
            mean_width = mean(width), 
            se_width = sd(width)/sqrt(n()), 
            mean_rmse = mean(rmse),
            se_rmse = sd(rmse)/sqrt(n())) 

# plots 

# not averaged over datasets
ggplot(data = combined_df %>% filter(traintest=='test'), 
       mapping = aes(x=mean_width, 
                     y=mean_coverage, 
                     color=factor(method)))+
  geom_point()+
  theme_bw()+
  xlab('Mean Test Set Width')+
  ylab('Mean Test Set Coverage')+
  geom_hline(yintercept = 0.95, color='black')+
  scale_color_discrete(name="Method",
                       breaks=c("dropout", "ensemble", "gp", "linear_regression", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "GP", "Linear Regression", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))

revised_fig <- ggplot(data = combined_df %>% filter(traintest=='test') %>%
         group_by(method) %>%
         summarise(sd_coverage = sd(mean_coverage), 
                   avg_coverage = mean(mean_coverage), 
                   avg_width=mean(mean_width), 
                   sd_width = sd(mean_width)), 
       mapping = aes(x=avg_width, 
                     y=avg_coverage, 
                     color=factor(method)))+
  geom_point()+
  geom_hline(yintercept = 0.95, color='black')+
  geom_errorbar(
    aes(ymin = avg_coverage-sd_coverage, ymax=avg_coverage+sd_coverage, color=factor(method)), 
    position = position_dodge(.3), width=0.1
  )+
  geom_errorbarh( aes(xmin = avg_width-sd_width, xmax=avg_width+sd_width, color=factor(method)), 
                   height=0.05)+
  theme_bw(base_size = 20)+
  xlab('Mean Test Set Width')+
  ylab('Mean Test Set Coverage')+
  scale_y_continuous(labels = scales::percent)+
  scale_color_discrete(name="Method",
                       breaks=c("dropout", "ensemble", "gp", "linear_regression", "ll_dropout", "ll_svi", "svi", "temp_scaling", 'vanilla'),
                       labels=c("Dropout", "Ensemble", "GP", "Linear Regression", "LL Dropout", "LL SVI", "SVI", "Temp Scaling", "Vanilla"))
  

inset <- ggplot(data = combined_df %>% filter(traintest=='test') %>%
         group_by(method) %>%
         summarise(sd_coverage = sd(mean_coverage), 
                   avg_coverage = mean(mean_coverage), 
                   avg_width=mean(mean_width), 
                   sd_width = sd(mean_width)), 
       mapping = aes(x=avg_width, 
                     y=avg_coverage, 
                     color=factor(method)))+
  geom_point()+
  geom_hline(yintercept = 0.95, color='black')+
  geom_errorbar(
    aes(ymin = avg_coverage-sd_coverage, ymax=avg_coverage+sd_coverage, color=factor(method)), 
    position = position_dodge(.3), width=0.1
  )+
  geom_errorbarh( aes(xmin = avg_width-sd_width, xmax=avg_width+sd_width, color=factor(method)), 
                  height=0.0022)+
  theme_bw(base_size = 20)+
  scale_y_continuous(labels = scales::percent, limits=c(.9,1))+
  xlab('Mean Test Set Width')+
  ylab('Mean Test Set Coverage')+
  theme(legend.position = "none")

rev_inset <- revised_fig + annotation_custom(ggplotGrob(inset), xmin=1.5, xmax=3, ymin=0.1, ymax=.75)

ggsave(paste0(base_dir, '/reports/figures/revised_coverage_width_no_inset.png'), plot = revised_fig, dpi = 300, width=7.5, height=3)
ggsave(paste0(base_dir, '/reports/figures/revised_coverage_width_inset.png'), plot = inset, dpi = 300, width=7.5, height=3)

ggsave(paste0(base_dir, '/reports/figures/revised_coverage_width.png'), plot = rev_inset, dpi = 300, width=7.5, height=3)


combined_df %>% filter(traintest=='test') %>% 
  select(dataset, method, mean_coverage, se_coverage) %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  pivot_wider(names_from=dataset, values_from = coverage) %>%
  write_csv(., './data/processed/regression_coverage_gp_temp.csv')

combined_df %>% filter(traintest=='test') %>% 
  select(dataset, method, mean_width, se_width) %>% 
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_width, se_width, sep=' ') %>% 
  pivot_wider(names_from=dataset, values_from = coverage) %>%
  write_csv(., './data/processed/regression_width_gp_temp.csv')

combined_df %>% filter(traintest=='test') %>% 
  select(dataset, method, mean_rmse, se_rmse) %>% 
  mutate(mean_rmse = round(mean_rmse,4)) %>%
  mutate(se_rmse = paste0("(", formatC(se_rmse,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_rmse, se_rmse, sep=' ') %>% 
  pivot_wider(names_from=dataset, values_from = coverage) %>%
  write_csv(., './data/processed/regression_rmse_gp_temp.csv')





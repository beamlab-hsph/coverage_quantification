library(tidyverse)
library(magrittr)

lr_df <- read_csv('/data/processed/lr.csv')
bayesian_df <- read_csv('/data/processed/model_output_summary_non_svi_eb.csv', 
                        col_names = T)
svi_df <- read_csv('/data/processed/model_output_summary_svi_eb.csv',
                   col_names = T)

gp_df <- read_csv('/data/processed/model_output_summary_gp.csv', col_names = T)

combined_df <- bind_rows(lr_df, bayesian_df)

combined_df %<>% bind_rows(combined_df, svi_df, gp_df)

combined_df %<>% group_by(dataset, method, traintest) %>% 
  summarise(mean_coverage = mean(coverage),
            se_coverage = sd(coverage)/sqrt(n()), 
            mean_width = mean(width), 
            se_width = sd(width)/sqrt(n()), 
            mean_rmse = mean(rmse),
            se_rmse = sd(rmse)/sqrt(n())) 

combined_df %>% filter(traintest=='test') %>% 
  select(dataset, method, mean_coverage, se_coverage) %>% 
  mutate(mean_coverage = round(mean_coverage,4)) %>%
  mutate(se_coverage = paste0("(", formatC(se_coverage,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_coverage, se_coverage, sep=' ') %>% 
  pivot_wider(names_from=dataset, values_from = coverage) %>%
  write_csv(., '/data/processed/regression_coverage_gp_temp.csv')

combined_df %>% filter(traintest=='test') %>% 
  select(dataset, method, mean_width, se_width) %>% 
  mutate(mean_width = round(mean_width,4)) %>%
  mutate(se_width = paste0("(", formatC(se_width,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_width, se_width, sep=' ') %>% 
  pivot_wider(names_from=dataset, values_from = coverage) %>%
  write_csv(., '/data/processed/regression_width_gp_temp.csv')

combined_df %>% filter(traintest=='test') %>% 
  select(dataset, method, mean_rmse, se_rmse) %>% 
  mutate(mean_rmse = round(mean_rmse,4)) %>%
  mutate(se_rmse = paste0("(", formatC(se_rmse,format="E", digits=2), ")")) %>%
  unite(col='coverage',mean_rmse, se_rmse, sep=' ') %>% 
  pivot_wider(names_from=dataset, values_from = coverage) %>%
  write_csv(., '/data/processed/regression_rmse_gp_temp.csv')





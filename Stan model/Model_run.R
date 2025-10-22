library(cmdstanr)
df = read.csv(file ='training_data_stanmod.csv')

dat = list(
  'S' = df$syllable,
  'L' = as.numeric(as.factor(df$lang)),
  'N' = nrow(df),
  'NL' = length(unique(df$lang)),
  'NP' = length(8:14),
  'dataMatrix' = as.matrix(df[,9:15])
)

file <- file.path('syll_newmodel_alterparam.stan')
mod <- cmdstan_model(file, pedantic = T)

fit <- mod$sample(
  data = dat, 
  chains = 4, 
  iter_warmup  = 2000,
  iter_sampling  = 2000,
  parallel_chains = 4,
  init = 0,
  refresh = 100
)


fit$save_output_files(basename = 'syll_newmodel_alterparam', timestamp = F, random = F)

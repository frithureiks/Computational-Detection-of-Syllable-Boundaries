library(cmdstanr)
library(bayesplot)
library(bayestestR)
library(boot)
library(ggplot2)
fit = as_cmdstan_fit(list.files()[grepl('syll_newmodel_alterparam-', list.files())])
fit$summary(variables = 'mus')


draws = fit$draws(variables = c('mus'), format = 'matrix')

plotdraws = draws[,-1]
colnames(plotdraws) <- c('Rpost', 'Rprev', 'Fin', 'Infprev', 'Infpost', 'Zeroprev', 'Zeropost')

plotdraws2 = data.frame()
for (col in 1:ncol(plotdraws)) {
  interv = bayestestR::hdi(plotdraws[,col], ci = 0.89)
  m = mean(plotdraws[,col])
  lower = interv[[3]]
  upper = interv[[4]]
  plotdraws2 = rbind(plotdraws2, data.frame('Parameter' = colnames(plotdraws)[col], m, lower, upper))
}

plotdraws3 = data.frame()
conveffects = c('Rpost', 'Rprev', 'Fin', 'Infprev', 'Infpost', 'Zeroprev', 'Zeropost')
langs = c("cs", "nl", "en", "fr", "de" ,"el", "it", "ko", "no", "es", "sv", "tr")
langs2 = c("Czech", "Dutch", "English", "French", "German" ,"Greek", "Italian", "Korean", "Norwegian", "Spanish", "Swedish", "Turkish")
for(i in 1:length(langs)){
  draws_tmp = data.frame(i = 1:8000)
  for (e in 2:8){
    draws_tmp = cbind(draws_tmp,as.data.frame(fit$draws(variables = paste0(c('effectsMatrix[',as.character(i),',', as.character(e), ']'), collapse = ''), format = 'matrix')))
  }
  draws_tmp = draws_tmp[,-1]
  for (col in 1:ncol(draws_tmp)) {
    interv = bayestestR::hdi(draws_tmp[,col], ci = 0.95)
    m = mean(draws_tmp[,col])
    lower = interv[[2]]
    upper = interv[[3]]
    plotdraws3 = rbind(plotdraws3, data.frame('Parameter' = conveffects[col], 'Language' = langs2[i], m, lower, upper))
  }
}
plotdraws2$Language = 'Hyperparameter'
plotdraws2 = plotdraws2[,c('Parameter', 'Language', 'm', 'lower', 'upper')]
plotdraws_big = rbind(plotdraws2,plotdraws3)
plotdraws_big$Language = factor(plotdraws_big$Language, c("Czech", "Dutch", "English", "French", "German" ,"Greek", 'Hyperparameter', "Italian", "Korean", "Norwegian", "Spanish", "Swedish", "Turkish"))
plotdraws_big$Parameter = factor(plotdraws_big$Parameter, levels = c('Rpost', 'Rprev', 'Fin', 'Infprev', 'Infpost', 'Zeroprev', 'Zeropost'))
g = ggplot() + geom_pointrange(plotdraws_big, mapping = aes(x = m,y = Parameter, xmin = lower, xmax = upper, color = Language), position = position_jitterdodge(dodge.width=0.8)) + 
  theme_classic() + scale_y_discrete(limits=rev) + xlab("Effect (logit)") + geom_vline(xintercept=c(0), linetype="dashed")+
  scale_color_manual(values=c("#F8766D", "#DE8C00", "#B79F00", "#7CAE00", "#00BA38", "#00C08B", 'black', "#00BFC4", "#00B4F0", "#619CFF", "#C77CFF",
                              "#F564E3", "#FF64B0"))+ 
  theme(legend.position="bottom")

pdf(file = 'effects_post2.pdf', width = 7, height = 8)
g
dev.off()



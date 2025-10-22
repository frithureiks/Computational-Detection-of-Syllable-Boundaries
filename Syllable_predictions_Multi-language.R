##########
#Load files for English and calculate metrics
##########
df1 = read.csv(file = 'LOO_English_Phone_500_FNN.csv')
df2 = read.csv(file = 'LOO_English_Phone_1500_FNN.csv')
df3 = read.csv(file = 'LOO_English_Phone_2500_FNN.csv')
df4 = read.csv(file = 'LOO_English_Phone_3500_FNN.csv')
df_full = read.csv(file = 'LOO_Results_Surprisal_Recalculated.csv')
df_full = df_full[df_full$Model == 'FNN' & df_full$Language == 'English',]

df1$size = '500'
df2$size = '1500'
df3$size = '2500'
df4$size = '3500'
df_full$size = 'full'
sup_df = rbind(df1[,c('Positions', 'Pred.Syllables', 'Ground.Truths', 'size')],
               df2[,c('Positions', 'Pred.Syllables', 'Ground.Truths', 'size')],
               df3[,c('Positions', 'Pred.Syllables', 'Ground.Truths', 'size')],
               df4[,c('Positions', 'Pred.Syllables', 'Ground.Truths', 'size')],
               data.frame('Positions' = df_full$Positions, 'Pred.Syllables' = df_full$Predictions, 'Ground.Truths' = df_full$Ground.Truths, 'size' = df_full$size))

res_df = data.frame()
for (pos in 2:11){
  for(size in unique(sup_df$size)){
    tmpdf = sup_df[sup_df$Positions == pos & sup_df$size == size,]
    TP = sum(tmpdf$Ground.Truths == 1 & tmpdf$Pred.Syllables == 1)
    FP = sum(tmpdf$Ground.Truths == 0 & tmpdf$Pred.Syllables == 1)
    FN = sum(tmpdf$Ground.Truths == 1 & tmpdf$Pred.Syllables == 0)
    precision = TP/(TP + FP)
    recall = TP/(TP+FN)
    res_df = rbind(res_df, data.frame('position' = pos, 'size' = size, 'precision' = precision, 'recall' = recall))
  }
}
res_rand = data.frame()
for (pos in 2:11){
  for(size in unique(sup_df$size)){
    tmpdf = sup_df[sup_df$Positions == pos & sup_df$size == size,]
    tmpdf$Pred.Syllables = rbinom(nrow(tmpdf), 1, .5)
    TP = sum(tmpdf$Ground.Truths == 1 & tmpdf$Pred.Syllables == 1)
    FP = sum(tmpdf$Ground.Truths == 0 & tmpdf$Pred.Syllables == 1)
    FN = sum(tmpdf$Ground.Truths == 1 & tmpdf$Pred.Syllables == 0)
    precision = TP/(TP + FP)
    recall = TP/(TP+FN)
    res_rand = rbind(res_rand, data.frame('position' = pos, 'size' = size, 'precision' = precision, 'recall' = recall))
  }
}
library(ggplot2)
res_df$size = factor(res_df$size, levels = c('500', '1500', '2500', '3500', 'full'))
res_rand$size = factor(res_rand$size, levels = c('500', '1500', '2500', '3500', 'full'))
res_df$f1 = 2 * (res_df$precision * res_df$recall) / (res_df$precision + res_df$recall)
res_rand$f1 = 2 * (res_rand$precision * res_rand$recall) / (res_rand$precision + res_rand$recall)

res_diff = res_df
res_diff[,3:ncol(res_diff)] = res_diff[,3:ncol(res_diff)] - res_rand[,3:ncol(res_rand)]
res_rand_minmax = data.frame(
  'position' = aggregate(precision ~ position, data = res_rand, FUN = max)[,1],
  'min' = aggregate(precision ~ position, data = res_rand, FUN = min)[,2],
  'max' = aggregate(precision ~ position, data = res_rand, FUN = max)[,2]
)

#plot results
ggplot() + geom_point(data = res_diff, mapping = aes(y = precision, x= position, color = size))+
  theme_classic()

##########
#Get syllable-level accuracies
##########
sup_df = rbind(df1,
               df2,
               df3,
               df4
               )


get_sylls <- function(string, sylls) {
  idxs = c(0, which(sylls == 1)[-1]-1,length(sylls))
  sylllist = c()
  for(i in 2:length(idxs)){
    sylllist = c(sylllist, paste0(string[(idxs[i-1]+1):idxs[i]], collapse = '') )
  }
  return(sylllist)
}

wordinds = sort(unique(sup_df$Word.Indices))
sizeinds = sort(unique(sup_df$size))
syll_true = data.frame()
syll_pred_onlyinit = data.frame()
syll_pred = data.frame()
len_wordinds = length(wordinds)
len_sizeinds = length(sizeinds)
for (i in 1:len_wordinds){
  print(i/len_wordinds)
  for (j in 1:len_sizeinds){
    tmpdf = sup_df[sup_df$Word.Indices == wordinds[i] & sup_df$size == sizeinds[j],]
    if(nrow(tmpdf) > 0 & sum(tmpdf$Ground.Truths) > 1 & sum(tmpdf$Pred.Syllables) > 1 ){
      pred_s = get_sylls(tmpdf$Segments, tmpdf$Pred.Syllables)
      true_s = get_sylls(tmpdf$Segments, tmpdf$Ground.Truths)
      print_size = sizeinds[j]
      print_language = tmpdf$Language[1]
      
      for(e in true_s){
        syll_true = rbind(syll_true, data.frame(
          'true' = e,
          'lang' = print_language,
          'size' = print_size
        ))
      }
      for(e in pred_s){
        syll_pred = rbind(syll_pred, data.frame(
          'pred' = e,
          'lang' = print_language,
          'size' = print_size
        ))
      }
      syll_pred_onlyinit = rbind(syll_pred_onlyinit, data.frame(
        'pred' = pred_s[1],
        'lang' = print_language,
        'size' = print_size
      ))
      
    }
  }
}


#evaluate the results

syll_pred = syll_pred_onlyinit

syll_true_total = read.csv('syll_true.csv')
syll_pred_total = read.csv('syll_pred_onlyinit.csv')

syll_pred = rbind(syll_pred, data.frame('pred' = syll_pred_total[syll_pred_total$model == 'FNN' & syll_pred_total$lang =='English','pred'],lang = 'English', 'size' = 'Total'))
syll_true = rbind(syll_true, data.frame('true' = syll_true_total[syll_true_total$model == 'FNN' & syll_true_total$lang =='English','true'],lang = 'English', 'size' = 'Total'))

acc_df = data.frame()
for(j in c(unique(sup_df$size), 'Total')){
  for(i in unique(sup_df$Language)){
    truetemp = syll_true[syll_true$lang == i &syll_true$size == j,]
    predtemp = syll_pred[syll_pred$lang == i &syll_pred$size == j,]
    realsylls = unique(truetemp$true)
    if(j == 'Total'){
      predsylls = sort(names(table(predtemp$pred )[table(predtemp$pred) > 10]))
    }else{
      predsylls = sort(names(table(predtemp$pred )))
    }
    precision = length(intersect(predsylls, realsylls))/length(predsylls)
    recall = length(intersect(predsylls, realsylls))/length(realsylls)
    f1 = 2*length(intersect(predsylls, realsylls))/(2*length(intersect(predsylls, realsylls)) + length(realsylls)-length(intersect(predsylls, realsylls)) + length(setdiff(realsylls,predsylls))) 
    acc_df = rbind(acc_df,data.frame(
      'size' = j,
      'language' = i,
      'length' = 'Total',
      'n' = length(predsylls),
      'precision' = precision,
      'recall' = recall,
      'f1' = f1
    ))
    for(e in 1:4){
      truetemp = syll_true[syll_true$lang == i & nchar(syll_true$true) == e &syll_true$size == j,]
      predtemp = syll_pred[syll_pred$lang == i & nchar(syll_pred$pred) == e &syll_pred$size == j,]
      realsylls = unique(truetemp$true)
      if(j == 'Total'){
        predsylls = sort(names(table(predtemp$pred )[table(predtemp$pred) > 10]))
      }else{
        predsylls = sort(names(table(predtemp$pred )))
      }
      precision = length(intersect(predsylls, realsylls))/length(predsylls)
      recall = length(intersect(predsylls, realsylls))/length(realsylls)
      f1 = 2*length(intersect(predsylls, realsylls))/(2*length(intersect(predsylls, realsylls)) + length(realsylls)-length(intersect(predsylls, realsylls)) + length(setdiff(realsylls,predsylls))) 
      acc_df = rbind(acc_df,data.frame(
        'size' = j,
        'language' = i,
        'length' = e,
        'n' = length(predsylls),
        'precision' = precision,
        'recall' = recall,
        'f1' = f1
      ))
    }
  }
}

print_df1 = acc_df[acc_df$length != 'Total',]
print_df1 = print_df1[!is.na(print_df1$precision),]
print_df1$size = factor(print_df1$size, levels = c('500', '1500', '2500', '3500', 'Total'))
print_df1$length = as.numeric(print_df1$length)
ggplot() + geom_line(data = print_df1, mapping = aes(y = precision, x= length, color = size), lwd = 1, alpha= .7)+geom_point(data = print_df1, mapping = aes(y = precision, x= length, group = size), pch = 21, color = 'black', fill = 'white', stroke = 1.5)+
  theme_classic() + ylim(0,1)


print_df1 = acc_df[acc_df$length != 'Total',]
print_df1 = print_df1[!is.na(print_df1$precision),]
print_df1$size = factor(print_df1$size, levels = rev(c('500', '1500', '2500', '3500', 'Total')))
print_df1$length = factor(print_df1$length)
ggplot() + geom_line(data = print_df1, mapping = aes(y = precision, x= size, color = length, group = length), lwd = 1, alpha= .7)+geom_point(data = print_df1, mapping = aes(y = precision, x= size, group = length), pch = 21, fill = 'white', stroke = 1.5)+
  theme_classic() + ylim(0,1)


aggsd_df = data.frame()
for(j in unique(acc_df$model)){
  for(i in unique(acc_df$length)){
    tmpdf = acc_df[acc_df$model == j & acc_df$length == i,]
    
    aggsd_df = rbind(aggsd_df,data.frame(
      'Metric' = 'precision',
      'Model' = j,
      'length' = i,
      'avg' = mean(tmpdf$precision, na.rm = T),
      'sd' = sd(tmpdf$precision, na.rm = T)
    ))
    aggsd_df = rbind(aggsd_df,data.frame(
      'Metric' = 'recall',
      'Model' = j,
      'length' = i,
      'avg' = mean(tmpdf$recall, na.rm = T),
      'sd' = sd(tmpdf$recall, na.rm = T)
    ))

  }   
}


aggsd_df_print = aggsd_df
aggsd_df_print$avg = paste0(as.character(round(aggsd_df_print$avg,digits = 3)), '  (' ,as.character(round(aggsd_df_print$sd,digits = 2)), ')')
aggsd_df_print = aggsd_df_print[,-5]
aggsd_df_print = reshape(aggsd_df_print, idvar = c("Model", 'Metric'), timevar = "length", direction = "wide")
aggsd_df_print = aggsd_df_print[order(aggsd_df_print$Metric),]


predtemp = syll_pred[syll_pred$lang == 'English' &syll_pred$model == 'FNN',]
truetemp = unique(syll_true[syll_true$lang == 'English' &syll_true$model == 'FNN', 'true'])
predsylls = sort(names(table(predtemp$pred )[table(predtemp$pred) > 10]))
predsylls = predsylls[nchar(predsylls)>1 &nchar(predsylls)<4]
trues = predsylls %in% truetemp

library(stringi)
uniqchars <- unique(unlist(strsplit(predsylls, "")))
predsylls2 <- stri_replace_all_regex(predsylls,
                                  pattern=c('_', 'Q', '\\{', 'R', '1', '2', '6', '\\$', '5', '\\#', 'V', '3' ),
                                  replacement=c('dZ', 'textturnscripta ', '\ae ', 'r', 'eI', 'aI', 'aU', 'O:', '@U', 'A:', 'textturnv ', 'textrevepsilon :'),
                                  vectorize=FALSE)
predsylls2 = gsub('^', ' textipa{', predsylls2)
predsylls2 = gsub('$', '}', predsylls2)

for ( i in 1:length(predsylls2)){
  if(!trues[i]){
    predsylls2[i] = gsub('^', '\\(', predsylls2[i])
    predsylls2[i] = gsub('$', '\\)', predsylls2[i])
  }
}



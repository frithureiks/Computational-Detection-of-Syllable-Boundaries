##########
#Load files for multi-language comparison and calculate metrics
##########

library(stringi)

sup_df = read.csv(file = 'LOO_Results_Original_NoHead_withbaselines.csv')

get_sylls <- function(string, sylls) {
  idxs = c(0, which(sylls == 1)[-1]-1,length(sylls))
  sylllist = c()
  for(i in 2:length(idxs)){
    sylllist = c(sylllist, paste0(string[(idxs[i-1]+1):idxs[i]], collapse = '') )
  }
  return(sylllist)
}

if (file.exists("syll_true.csv")) {
  file.remove("syll_true.csv")
}
if (file.exists("syll_pred.csv")) {
  file.remove("syll_pred.csv")
}
if (file.exists("syll_pred_onlyinit.csv")) {
  file.remove("syll_pred_onlyinit.csv")
}

write(paste0(c('true', 'lang', 'model'), collapse = ','),file="syll_true.csv",append=TRUE)
write(paste0(c('pred', 'lang', 'model'), collapse = ','),file="syll_pred.csv",append=TRUE)
write(paste0(c('pred', 'lang', 'model'), collapse = ','),file="syll_pred_onlyinit.csv",append=TRUE)

wordinds = sort(unique(sup_df$Word.Indices))
modelinds = sort(unique(sup_df$Model))
syll_true = data.frame()
syll_pred_onlyinit = data.frame()
syll_pred = data.frame()
len_wordinds = length(wordinds)
len_modelinds = length(modelinds)
for (i in 1:len_wordinds){
  print(i/len_wordinds)
  for (j in 1:len_modelinds){
    tmpdf = sup_df[sup_df$Word.Indices == wordinds[i] & sup_df$Model == modelinds[j],]
    pred_s = get_sylls(tmpdf$Segments, tmpdf$Predictions)
    true_s = get_sylls(tmpdf$Segments, tmpdf$Ground.Truths)
    print_model = modelinds[j]
    print_language = tmpdf$Language[1]
    
    write.table(data.frame(true_s, print_language, print_model), file = "syll_true.csv", sep = ",",
                append = TRUE, quote = FALSE,
                col.names = FALSE, row.names = FALSE)
    
    write.table(data.frame(pred_s, print_language, print_model), file = "syll_pred.csv", sep = ",",
                append = TRUE, quote = FALSE,
                col.names = FALSE, row.names = FALSE)
    
    write.table(data.frame(pred_s[1], print_language, print_model), file = "syll_pred_onlyinit.csv", sep = ",",
                append = TRUE, quote = FALSE,
                col.names = FALSE, row.names = FALSE)
  
  }
}

syll_true = read.csv('syll_true.csv')
syll_pred_onlyinit = read.csv('syll_pred_onlyinit.csv')

sup_df = read.csv(file = 'LOO_Results_Original_NoHead_withbaselines.csv')
syll_pred = syll_pred_onlyinit

acc_df = data.frame()
for(j in unique(sup_df$Model)){
  for(i in unique(sup_df$Language)){
    truetemp = syll_true[syll_true$lang == i &syll_true$model == j,]
    predtemp = syll_pred[syll_pred$lang == i &syll_pred$model == j,]
    realsylls = unique(truetemp$true)
    predsylls = sort(names(table(predtemp$pred )[table(predtemp$pred) > 10]))
    precision = length(intersect(predsylls, realsylls))/length(predsylls)
    recall = length(intersect(predsylls, realsylls))/length(realsylls)
    f1 = 2*length(intersect(predsylls, realsylls))/(2*length(intersect(predsylls, realsylls)) + length(realsylls)-length(intersect(predsylls, realsylls)) + length(setdiff(realsylls,predsylls))) 
    acc_df = rbind(acc_df,data.frame(
      'model' = j,
      'language' = i,
      'length' = 'Total',
      'n' = length(predsylls),
      'precision' = precision,
      'recall' = recall,
      'f1' = f1
    ))
    for(e in 1:4){
      truetemp = syll_true[syll_true$lang == i & nchar(syll_true$true) == e &syll_true$model == j,]
      predtemp = syll_pred[syll_pred$lang == i & nchar(syll_pred$pred) == e &syll_pred$model == j,]
      realsylls = unique(truetemp$true)
      predsylls = sort(names(table(predtemp$pred )[table(predtemp$pred) > 10]))
      precision = length(intersect(predsylls, realsylls))/length(predsylls)
      recall = length(intersect(predsylls, realsylls))/length(realsylls)
      f1 = 2*length(intersect(predsylls, realsylls))/(2*length(intersect(predsylls, realsylls)) + length(realsylls)-length(intersect(predsylls, realsylls)) + length(setdiff(realsylls,predsylls))) 
      acc_df = rbind(acc_df,data.frame(
        'model' = j,
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


predtemp = syll_pred[syll_pred$lang == 'English' &syll_pred$model == 'CNN',]
truetemp = unique(syll_true[syll_true$lang == 'English' &syll_true$model == 'CNN', 'true'])
predsylls = sort(names(table(predtemp$pred )[table(predtemp$pred) > 10]))
predsylls = predsylls[nchar(predsylls)>1 &nchar(predsylls)<4]
trues = predsylls %in% truetemp



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




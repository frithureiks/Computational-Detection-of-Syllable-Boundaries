df = read.csv(file="All_Languages.csv", header = T)
df = df[,-1]
df$r_prev = NA
df$r_post = NA
df$fin = 0
df$infprev = 0
df$infpost = 0
df$zeroprev = 0
df$zeropost = 0
df2 = data.frame()
uniqueids = unique(df$word_id)
N = length(uniqueids)
for (i in 1:N){
  if( i %% 1000 == 0 ) cat(paste(round(i/N, digits = 3)))
  w = uniqueids[i]
  tmp = df[df$word_id == w,]
  if(nrow(tmp)> 1){
    tmp$r_prev[2:(nrow(tmp))] = tmp$surprisal[2:(nrow(tmp))]/tmp$surprisal[1:(nrow(tmp)-1)]
    tmp$r_post[1:(nrow(tmp)-1)] = tmp$surprisal[1:(nrow(tmp)-1)]/tmp$surprisal[2:(nrow(tmp))]
    tmp[which(tmp$r_prev == Inf), 'infprev'] = 1
    tmp[which(tmp$r_post == Inf), 'infpost'] = 1
    tmp[which(tmp$r_prev == 0), 'zeroprev'] = 1
    tmp[which(tmp$r_post == 0), 'zeropost'] = 1
    tmp[nrow(tmp), 'fin'] = 1
    df2 = rbind(df2, tmp[rowSums(is.na(tmp[,7:8]))<2,][-1,])
  }
}

write.csv(df2, file ='full_data_stanmod.csv')

set.seed(456)
df = read.csv('full_data_stanmod.csv')

df$r_prev = df$r_prev*(1-df$infprev)*(1-df$zeroprev)
df$r_post = df$r_post*(1-df$fin)*(1-df$infpost)*(1-df$zeropost)

df$r_prev[is.na(df$r_prev)] = 0
df$r_post[is.na(df$r_post)] = 0

df = df[-which(df$r_prev > 10 | df$r_post > 10),]

df$r_prev[df$r_prev!=0] = as.vector(scale(df$r_prev[df$r_prev!=0], center = F))
df$r_post[df$r_post!=0] = as.vector(scale(df$r_post[df$r_post!=0], center = F))

df = df[!duplicated(df[,c(-1,-2)]),]

testidxs = c()
for(i in unique(df$lang)){
  tmpdf = df[df$lang == i,]
  tmpdf = tmpdf[!duplicated(tmpdf$word),]
  tmpdf = tmpdf[sample(1:nrow(tmpdf), 3000),]
  testidxs = c(testidxs, which(df$word %in% tmpdf$word))
  
}

test_df = df[testidxs,]
df = df[-testidxs,]

write.csv(df, file ='training_data_stanmod.csv')
write.csv(test_df, file ='test_data_stanmod.csv')
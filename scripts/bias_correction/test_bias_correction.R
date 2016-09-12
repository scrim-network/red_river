# TODO: Add comment
# 
# Author: jwo118
###############################################################################
Sys.setenv(TZ="UTC")
library(xts)

source('/storage/home/jwo118/projects/red_river/esd_r/debias.R')


################################################################################
# Test Tair
################################################################################

df <- read.csv('/storage/home/jwo118/group_store/red_river/bias_correction/test_bias_data.csv', stringsAsFactors=F)
df <- xts(df[,c('obs','model')], as.POSIXct(df$time))
idx_train <- '1976/2005'
idx_fut <- c('1976/2005','2006/2039','2040/2069','2070/2099')


mod_adj <- mw_bias_correct(df$obs, df$model, idx_train, idx_fut, edqmap_tair)
ann_adj <- apply.yearly(apply.monthly(mod_adj,mean),mean)['2006/2099']
ann <- apply.yearly(apply.monthly(df$model,mean),mean)['2006/2099']
ann_anoms_adj <- ann_adj - mean(ann_adj)
ann_anoms <- ann - mean(ann)

plot(ann_anoms)
lines(ann_anoms_adj,col='red')

mod_adj_qm <- mw_bias_correct(df$obs, df$model, idx_train, c('1976/2099'), qmap_tair)
ann_adj_qm <- apply.yearly(apply.monthly(mod_adj_qm,mean),mean)['2006/2099']
ann_anoms_adj_qm <- ann_adj_qm - mean(ann_adj_qm)

################################################################################
# Plots

# Annual Anomalies
plot(ann_anoms,main='Annual Anomalies', ylab='degC')
lines(ann_anoms_adj, col='red')
lines(ann_anoms_adj_qm, col='blue')
legend('topleft',c('Orig','QM','EDQM'),col=c('black','blue','red'),lty=1)

# Anomaly Differences
plot(rollapply(ann_anoms_adj_qm-ann_anoms,5,mean),main='Annual Anomaly Differences\n5-year Running Mean',ylab='degC')
lines(rollapply(ann_anoms_adj-ann_anoms,5,mean),col='red')
lines(rollapply(ann_anoms_adj-ann_anoms_adj_qm,5,mean),col='blue')
legend('top',c('QM Minus Orig','EDQM Minus Orig','EDQM Minus QM'),col=c('black','red','blue'),lty=1,cex=.75)

# CDF plot: 1976-2005 Model vs Obs
plot(ecdf(as.numeric(df$obs['1976/2005'])),verticals=T,main='1976-2005 CDF',xlab='degC',ylab='Cumulative Probability')
lines(ecdf(as.numeric(df$model['1976/2005'])),verticals=T,main='1976-2005 CDF',xlab='degC',ylab='Cumulative Probability',col='red')
legend('topleft',c('Observed','CCSM4'),col=c('black','red'),lty=1)

# CDF plot: 2070-2099 EDQM vs QM
plot(ecdf(as.numeric(mod_adj['2070/2099'])),verticals=T,main='2070-2099 CDF',xlab='degC',ylab='Cumulative Probability',col='red')
lines(ecdf(as.numeric(mod_adj_qm['2070/2099'])),verticals=T,col='blue')
legend('topleft',c('QM','EDQM'),col=c('blue','red'),lty=1)
# Dif Future
q <- ecdf(as.numeric(mod_adj_qm['2070/2099']))(sort(as.numeric(mod_adj_qm['2070/2099'])))
plot(sort(as.numeric(mod_adj_qm['2070/2099']))-sort(as.numeric(mod_adj['2070/2099'])),q,type='l',col='black',main='2070-2099 CDF Difference\nQM Minus EDQM',xlab='degC',ylab='Cumulative Probability')
abline(v=0,col='gray')

# Dif Baseline
q <- ecdf(as.numeric(df$obs['1976/2005']))(sort(as.numeric(df$obs['1976/2005'])))
plot(sort(as.numeric(a['1976/2005']))-sort(as.numeric(df$obs['1976/2005'])),q,type='l',col='black',main='1979-2005 CDF Difference\nEDQM Minus Obs',xlab='degC',ylab='Cumulative Probability')
abline(v=0,col='gray')


################################################################################
# Test Prcp
################################################################################

df <- read.csv('/storage/home/jwo118/group_store/red_river/bias_correction/test_bias_data_prcp_gfdl.csv', stringsAsFactors=F)
df <- xts(df[,c('obs','model')], as.POSIXct(df$time))
idx_train <- '1976/2005'
idx_fut <- c('1976/2005','2006/2039','2040/2069','2070/2099')
idx_all <- unique(c(idx_train,idx_fut))
winsize = 31

# Plot monthly normals
plot(sapply(0:11,function(x){mean(df$mod['1976/2005'][.indexmon(df$mod['1976/2005'])==x])}),type='l',col='red')
lines(sapply(0:11,function(x){mean(df$obs['1976/2005'][.indexmon(df$obs['1976/2005'])==x])}),type='l')

# Correcting for artificially inflated extremes in moving window correction
################################################################################
win_masks <- build_window_masks(df,idx_all,winsize=winsize)
win_masks1 <- build_window_masks(df,idx_all,winsize=1)

mod_adj <- mw_bias_correct(df$obs,df$mod,idx_train,c('2070/2099'),edqmap_prcp,win_masks,win_masks1,correct_win=FALSE)
mod_adj_nowin <- edqmap_prcp(df$obs['1976/2005'], df$mod['1976/2005'], df$mod['2070/2099'], df$mod['2070/2099'])

mod_adj <- mw_bias_correct(df$obs,df$mod,idx_train,c('2070/2099'),edqmap_prcp,winsize=winsize)



# QQPlot: mod_adj vs. mod_adj_nowin
qqplot(as.numeric(mod_adj_nowin),as.numeric(mod_adj),xlab='No window (mm)',ylab='31-day window (mm)',main='Bias Corrected Prcp 2070-2099')
abline(0,1,col='red')

# Compare monthly normals
plot(sapply(0:11,function(x){mean(mod_adj_nowin[.indexmon(mod_adj_nowin)==x])}),type='l')
lines(sapply(0:11,function(x){mean(mod_adj[.indexmon(mod_adj)==x])}),type='l',col='red')

# Bias correct mod_adj with mod_adj_nowin
mod_adj2 <- edqmap_prcp(mod_adj_nowin, mod_adj, mod_adj, mod_adj)


ann <- apply.yearly(apply.monthly(df$model,sum),sum)['2006/2099']
ann_anoms <- ann/mean(ann)


mod_adj_qm <- mw_bias_correct(df$obs, df$model, idx_train, c('1976/2099'), qmap_tair)
ann_adj_qm <- apply.yearly(apply.monthly(mod_adj_qm,sum),sum)['2006/2099']
ann_anoms_adj_qm <- ann_adj_qm/mean(ann_adj_qm)



# Test EDQMAP for single window
idx_train <- '1976/2005'
a_idx_fut <- '2070/2099'#'2070/2099'
winsize = 31
obs <- df$obs
mod <- df$mod

#a <- mw_bias_correct(obs,mod,idx_train,c(a_idx_fut),edqmap_prcp,winsize=winsize)
#b <- mw_bias_correct(obs,mod,idx_train,c(a_idx_fut),edqmap_prcp_pierce)

masks_win_train <- window_masks(index(obs[idx_train]), winsize = winsize)
masks_win_fut <- window_masks(index(mod[a_idx_fut]), winsize = winsize)
masks_mthday_fut <- window_masks(index(mod[a_idx_fut]),1)

mask_win_train = masks_win_train['01-04',]
mask_win_fut <- masks_win_fut['01-04',]
mask_day_fut <- masks_mthday_fut['01-04',]

obs <- obs[idx_train][mask_win_train]
pred_train <- mod[idx_train][mask_win_train]
pred_fut <- mod[a_idx_fut][mask_win_fut]
pred_fut_subset <- mod[a_idx_fut][mask_day_fut]

pred_fut_subset <- pred_fut

win_masks <- build_window_masks(df$obs, unique(c(idx_train,idx_fut)), 31)
win_masks1 <- build_window_masks(df$obs, unique(c(idx_train,idx_fut)), 1)


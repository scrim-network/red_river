# TODO: Add comment
# 
# Author: jwo118
###############################################################################
Sys.setenv(TZ="UTC")
library(xts)

window_masks <- function(dates, winsize=31) {

  delta <- as.difftime(round((winsize-1)/2), units='days')
  mth_day <- strftime(dates,'%m-%d')
  
  dates_yr <- seq(as.POSIXlt('2007-01-01'), as.POSIXlt('2007-12-31'), 'day')
  
  win_masks <- matrix(NA,nrow=length(dates_yr),ncol=length(dates))
  row.names(win_masks) <- strftime(dates_yr,'%m-%d')
  
  for (i in seq(length(dates_yr))) {
    a_date <- dates_yr[i]
    mth_day_win <- strftime(seq(a_date - delta, a_date + delta, 'day'),'%m-%d')
    win_masks[i,] <- mth_day %in% mth_day_win
  }
  
  # Add month day win for leap day
  a_date <- as.POSIXct('2008-02-29')
  mth_day_win <- strftime(seq(a_date - delta, a_date + delta, 'day'),'%m-%d')
  mask_leap <- mth_day %in% mth_day_win
  mask_leap <- matrix(mask_leap,nrow=1,ncol=length(mask_leap))
  row.names(mask_leap) <- '02-29'
  
  i28 <- which(row.names(win_masks) == '02-28')
  
  win_masks <- rbind(win_masks[1:i28,],
                     mask_leap,
                     win_masks[(i28+1):nrow(win_masks),])
                 
  return(win_masks)

}

build_window_masks <- function(obs, idx_periods, winsize=31) {
  
  win_masks <- list()
    
  for (a_idx in idx_periods) {
  
    win_masks[[a_idx]] <- window_masks(index(obs[a_idx]), winsize = winsize)
  
  }
  
  return(win_masks)
  
}

mw_bias_correct <- function(obs, mod, idx_train, idx_fut, bias_func, win_masks, win_masks_1day) {
    
  mod_fut_adj <- NULL
  
  masks_win_train <- win_masks[[idx_train]]
  
  for (a_idx_fut in idx_fut) {
    
    masks_win_fut <- win_masks[[a_idx_fut]]
    masks_mthday_fut <- win_masks_1day[[a_idx_fut]]
    
    vals_obs <- obs[idx_train]
    vals_mod_train <- mod[idx_train]
    vals_mod_fut <- mod[a_idx_fut]
    vals_mod_fut_adj <- xts(vals_mod_fut)
    
    xtsAttributes(vals_mod_train)$modname <- paste(xtsAttributes(vals_mod_train)$modname,a_idx_fut,sep='_')
    xtsAttributes(vals_mod_fut_adj) <- xtsAttributes(vals_mod_train)
    
    for (a_day in row.names(masks_mthday_fut)) {   
      
      mask_win_train <- masks_win_train[a_day,]
      mask_win_fut <- masks_win_fut[a_day,]
      mask_day_fut <- masks_mthday_fut[a_day,]
            
      vals_mod_fut_adj[mask_day_fut] <- bias_func(vals_obs[mask_win_train],
                                                  vals_mod_train[mask_win_train],
                                                  vals_mod_fut[mask_win_fut],
                                                  vals_mod_fut[mask_day_fut])
    }
    
    vals_mod_fut_adj_nowin <- bias_func(vals_obs, vals_mod_train, vals_mod_fut, vals_mod_fut) 
    
    # Bias correct vals_mod_fut_adj with vals_mod_fut_adj_nowin
    vals_mod_fut_adj <- bias_func(vals_mod_fut_adj_nowin, vals_mod_fut_adj, vals_mod_fut_adj, vals_mod_fut_adj)
    
    mod_fut_adj <- rbind(mod_fut_adj, vals_mod_fut_adj)
    
  }
  
  return(mod_fut_adj)
  
}

# Based on eqm function in downscaleR
# https://github.com/SantanderMetGroup/downscaleR/blob/devel/R/biasCorrection.R
qmap_vals <- function(from_vals_train, to_vals_train, from_vals) {
  
  n <- min(c(length(from_vals_train),c(length(to_vals_train))))  
  q_step <- 1/(n-1)
  probs <- seq(0,1,q_step)
  
  q_from <- quantile(from_vals_train, probs = probs)
  q_to <- quantile(to_vals_train, probs = probs)
  
  qm_func <- approxfun(q_from, q_to, method = "linear")
  
  vals_mapped <- qm_func(from_vals)
  
  # Extrapolate differences of smallest/larget quantiles to map values in to_vals
  # that are outside the range of those in to_vals_train
  mask_above <- which(from_vals > max(q_from))
  mask_below <- which(from_vals < min(q_from))
  
  vals_mapped[mask_above] <- from_vals[mask_above] + (q_to[length(q_to)] - q_from[length(q_from)])
  vals_mapped[mask_below] <- from_vals[mask_below] + (q_to[1] - q_from[1])
  
  return(vals_mapped)

}


# Delta type: 'add' or 'ratio'
edqmap <- function(obs, pred_train, pred_fut, pred_fut_subset, delta_type='add') {
  
  delta_type <- match.arg(delta_type, c('add','ratio'))
  
  ##############################################################################
  # Map pred_fut to pred_train and calculate deltas at each quantile
  ##############################################################################    
  mapped_train <- qmap_vals(pred_fut,pred_train,pred_fut_subset)
  
  if (delta_type == 'add') {
    delta <- pred_fut_subset-mapped_train
  } 
  else if (delta_type == 'ratio') {
    delta <- pred_fut_subset/mapped_train
    
    # If mapped_train is 0, delta will be Inf or NA. Set these to 1
    delta[is.infinite(delta) | is.na(delta)] = 1
    
    #TODO: how to avoid extremely large deltas?
       
  }
  
  ##############################################################################
  
  ##############################################################################
  # Map pred_fut to obs and add deltas
  ##############################################################################  
  mapped_obs <- qmap_vals(pred_fut, obs, pred_fut_subset) 
  
  if (delta_type == 'add') {
    pred_fut_adj <- mapped_obs + delta
  } 
  else if (delta_type == 'ratio') {
    pred_fut_adj <- mapped_obs * delta
  }
  
  ############################################################################## 
    
  return(pred_fut_adj)
  
}


edqmap_prcp <- function(obs, pred_train, pred_fut, pred_fut_subset) {
  
  wetday_t <- wetday_threshold(obs, pred_train)
  pred_train0 <- pred_train
  pred_fut0 <- pred_fut
  pred_fut_subset0 <- pred_fut_subset
  
  pred_train0[pred_train0 < wetday_t] = 0
  pred_fut0[pred_fut0 < wetday_t] = 0
  pred_fut_subset0[pred_fut_subset0 < wetday_t] = 0
  
  obs_wet <- obs[obs > 0]
  pred_train_wet  <- pred_train0[pred_train0 > 0]
  pred_fut_wet  <- pred_fut0[pred_fut0 > 0]
  pred_fut_subset_wet  <- pred_fut_subset0[pred_fut_subset0 > 0]
  
  pred_fut_adj_wet <- edqmap(obs_wet, pred_train_wet, pred_fut_wet, pred_fut_wet, delta_type='ratio')
  pred_fut_adj <- pred_fut0
  pred_fut_adj[pred_fut_adj > 0] = as.numeric(pred_fut_adj_wet)
  
  # Change Factor Correction
  x <- mean(pred_fut0)/mean(pred_train0)
  
  # Bias correct base training period
  pred_train_adj_wet <- edqmap(obs_wet, pred_train_wet, pred_train_wet, pred_train_wet, delta_type='ratio')
  pred_train_adj <- pred_train0
  pred_train_adj[pred_train_adj > 0] = as.numeric(pred_train_adj_wet)
  
  xhat <- mean(mean(pred_fut_adj)/mean(pred_train_adj))
  
  k <- x/xhat
    
  pred_fut_adj <- pred_fut_adj*k
  
  return(pred_fut_adj[index(pred_fut_subset)])
}

# threshold below which mod prcp should be set to 0
wetday_threshold <- function(obs, mod) {
  
  nwet_obs <- sum(obs > 0)
  nwet_mod <- sum(mod > 0)
  
  ndif <- nwet_mod-nwet_obs
  
  if (ndif > 0) {
    thres <- sort(as.numeric(mod[mod > 0]))[ndif+1]
  } else {
    thres <- 0
    
    if (ndif != 0) {
      
      #cat(sprintf("Warning: %s has dry bias of %d days over %d total days from %s-%s. \n", xtsAttributes(mod)$modname, ndif,length(obs),strftime(min(index(obs)),'%Y'),strftime(max(index(obs)),'%Y')), file=fpath_log, append=TRUE)
      
      print(sprintf("Warning: %s has dry bias of %d days over %d total days from %s-%s", xtsAttributes(mod)$modname, ndif,length(obs),strftime(min(index(obs)),'%Y'),strftime(max(index(obs)),'%Y')))
    }

  }
 
  return(thres)
  
}

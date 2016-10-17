# Functions for GCM bias correction
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

mw_bias_correct <- function(obs, mod, idx_train, idx_fut, bias_func, win_masks,
                            win_masks_1day, correct_win=TRUE) {
    
  mod_fut_adj <- NULL
  
  masks_win_train <- win_masks[[idx_train]]
  
  for (a_idx_fut in idx_fut) {
    
    masks_win_fut <- win_masks[[a_idx_fut]]
    masks_mthday_fut <- win_masks_1day[[a_idx_fut]]
    
    vals_obs <- obs[idx_train]
    vals_mod_train <- mod[idx_train]
    vals_mod_fut <- mod[a_idx_fut]
    vals_mod_fut_adj <- xts(vals_mod_fut)
    
    xtsAttributes(vals_mod_train)$modname <- paste(xtsAttributes(vals_mod_train)$modname,
                                                   a_idx_fut,sep='_')
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
    
    if (correct_win) {
      
      vals_mod_fut_adj_nowin <- bias_func(vals_obs, vals_mod_train, vals_mod_fut, vals_mod_fut) 
      
      # Bias correct vals_mod_fut_adj with vals_mod_fut_adj_nowin
      vals_mod_fut_adj <- bias_func(vals_mod_fut_adj_nowin, vals_mod_fut_adj,
                                    vals_mod_fut_adj, vals_mod_fut_adj)
    }

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
  } else if (delta_type == 'ratio') {
    delta <- pred_fut_subset/mapped_train
    
    # If mapped_train is 0, delta will be Inf or NA. Set these to 1
    delta[is.infinite(delta) | is.na(delta)] = 1
    
    #TODO: Check for extremely large deltas?
#    if (length(unique(delta)) > 1)
#    {
#      # Limit delta by 6 standard deviations to avoid extremely inflated deltas
#      zdelta <- scale(delta)
#      max_delta <- max(delta[zdelta <= 6])
#      min_delta <- min(delta[zdelta >= -6])
#      delta[zdelta > 6] = max_delta
#      delta[zdelta < -6] = min_delta
#    }
         
  }
  
  ##############################################################################
  
  ##############################################################################
  # Map pred_fut to obs and add deltas
  ##############################################################################  
  mapped_obs <- qmap_vals(pred_fut, obs, pred_fut_subset) 
  
  if (delta_type == 'add') {
    pred_fut_adj <- mapped_obs + delta
  } else if (delta_type == 'ratio') {
    pred_fut_adj <- mapped_obs * delta
  }
  
  ############################################################################## 
    
  return(pred_fut_adj)
  
}

wetday_bias <- function(obs, pred_train, pred_fut, pred_fut_subset) {
  
  nwet_obs <- sum(obs > 0)
  nwet_mod <- sum(pred_train > 0)
    
  wet_bias <- pred_fut_subset
  wet_bias[] <- nwet_mod/nwet_obs

  return(wet_bias)
  
}

# threshold below which mod prcp should be set to 0
wetday_threshold <- function(obs, mod) {
  
  nwet_obs <- sum(obs > 0)
  nwet_mod <- sum(mod > 0)
  
  ndif <- nwet_mod-nwet_obs
  
  if (ndif > 0) {
    thres <- sort(as.numeric(mod[mod > 0]))[ndif]
    
    if (sum(mod > thres) != nwet_obs) {
      print(sprintf("Warning: %s has inexact prcp threshold. With threshold, ndry modeled = %d. ndry obs = %d",
                    xtsAttributes(mod)$modname,sum(mod <= thres),sum(obs == 0)))
    }
    
  } else {
    thres <- 0
    
#    if (ndif != 0) {
#      
#      print(sprintf("Warning: %s has dry bias of %d days over %d total days from %s-%s",
#                    xtsAttributes(mod)$modname, ndif,length(obs),strftime(min(index(obs)),'%Y'),
#                    strftime(max(index(obs)),'%Y')))
#    }

  }
  return(thres)
  
}

.check_prcp_kmean <- function(kmean, modname) {
  
  if (is.na(kmean) | is.infinite(kmean)) { kmean <- 1 }
  
}

edqmap_prcp <- function(obs, mod_train, mod_fut, mod_fut_subset) {
  
  # Calculate wet day threshold for bias correcting the model's # of wet days
  wetday_t <- wetday_threshold(obs, mod_train)
  # Create new model training and future time series setting all prcp values
  # <= to the wet day threshold to 0
  mod_train0 <- mod_train
  mod_fut0 <- mod_fut
  mod_train0[mod_train0 <= wetday_t] = 0
  mod_fut0[mod_fut0 <= wetday_t] = 0
  
  # Calculate mean of observation and model time series over training time period
  # and mean of model time series over future period
  mean_obs <- mean(obs)
  mean_mod_train0 <- mean(mod_train0)
  mean_mod_fut0 <- mean(mod_fut0)
  # Calculate ratio of observation mean to model mean.
  mean_bc_delta0 <- mean_obs/mean_mod_train0
    
  # Subset anomly time series to only wet values. Will only apply EDCDFm to wet
  # day values
  obs_wet <- obs[obs > 0]
  mod_train0_wet <- mod_train0[mod_train0 > 0]
  mod_fut0_wet <- mod_fut0[mod_fut0 > 0]
  
  # Calculate mean of observation and model time series over training time period
  # and mean of model time series over future period for just wet days
  mean_obs_wet <- mean(obs_wet)
  mean_mod_train0_wet <- mean(mod_train0_wet)
  mean_mod_fut0_wet <- mean(mod_fut0_wet)
  # Calculate ratio of observation mean to model mean for wet days.
  mean_bc_delta_wet <- mean_obs_wet/mean_mod_train0_wet
  
  # Convert all wet day time series to ratio anomalies based on their mean wet value
  anom_obs_wet <- obs_wet/mean_obs_wet
  anom_mod_train0_wet <- mod_train0_wet/mean_mod_train0_wet
  anom_mod_fut0_wet <- mod_fut0_wet/mean_mod_fut0_wet
  
  # Only apply EDCDFm if there are enough wet days in the time series
  no_wet <- (length(anom_obs_wet) <= 1 |
             length(anom_mod_train0_wet) <= 1 |
             length(anom_mod_fut0_wet) <= 1)
  
  if (no_wet) {
    
    # Not enough wet days to run EDCDFm. Simply set adjusted future wet time series
    # to the original anomaly values
    anom_mod_fut_adj_wet <-  anom_mod_fut0_wet
    
    print(sprintf("Error: Not enough wet days to bias correct %s. Will not bias correct.",
                  as.character(attributes(mod_train)$modname)))
  } else {
    
    # Run EDCDFm to bias correct future wet day anomalies
    anom_mod_fut_adj_wet <- edqmap(anom_obs_wet, anom_mod_train0_wet, anom_mod_fut0_wet,
                                   anom_mod_fut0_wet, delta_type='add')
    
    # If deltas applied by EDCDFm are significantly skewed, mean of ratio
    # anomalies might be slightly different from 1. Calculate k correction factor
    # to bring the ratio anomaly mean to 1. This will maintain the GCM projected
    # change in the mean wet day value. This k correction is analogous to the k correction
    # in Pierce et al. Section 3b
    anom_kmean <- check_prcp_kmean(1/mean(anom_mod_fut_adj_wet))
    
    # Calculate bias-corrected mean value for future time period
    mean_mod_fut_wet_adj <- mean_bc_delta_wet*mean_mod_fut0_wet
    
    # Multiply bias corrected daily anomaly ratios by k and bias-corrected mean
    # to get bias corrected daily wet day values
    mod_fut_adj_wet <- anom_mod_fut_adj_wet * anom_kmean * mean_mod_fut_wet_adj
  
  }
  
  # Combine dry days and adjusted wet days of future model time series to get
  # full adjusted time series
  mod_fut_adj <- mod_fut0
  mod_fut_adj[mod_fut_adj > 0] = as.numeric(mod_fut_adj_wet)
  
  # TODO: Should this correction be applied? It corrects the overall mean, but
  # then the wet day mean is no longer correct. In other words, which should have
  # higher priority--wet day mean or overall mean?
  
  # Make sure overall mean of mod_fut_adj matches the bias corrected overall mean
  # They should already be identical if the wet day threshold made the model # of wet 
  # days = # of observed wet days over the training period. If not (i.e.--the model has too
  # many dry days), the mean of mod_fut_adj will != the bias corrected overall mean
  # without an adjustment.
  
  # Calculate overall bias-corrected mean value for future time period 
  mean_mod_fut0_adj <- mean_bc_delta0*mean_mod_fut0
  # Calculate k correction factor ann apply to mod_fut_adj
  kmean <- check_prcp_kmean(mean_mod_fut0_adj/mean(mod_fut_adj))
  mod_fut_adj <- mod_fut_adj * kmean
  
  return(mod_fut_adj[index(mod_fut_subset)])
}

edqmap_tair <- function(obs, mod_train, mod_fut, mod_fut_subset) {
    
  # Calculate mean of observation and model time series over training time period
  # and mean of model time series over future period
  mean_obs <- mean(obs)
  mean_mod_train <- mean(mod_train)
  mean_mod_fut <- mean(mod_fut)
  # Calculate difference of observation mean to model mean over training period
  mean_bc_delta <- mean_obs - mean_mod_train
  
  # Calculate daily anomalies for each time series based on mean values
  anom_obs <- obs - mean_obs
  anom_mod_train <- mod_train - mean_mod_train
  anom_mod_fut <- mod_fut - mean_mod_fut
  
  # Run EDCDFm to bias correct future anomalies
  anom_mod_fut_adj <- edqmap(anom_obs, anom_mod_train, anom_mod_fut, anom_mod_fut, delta_type='add')
    
  # If deltas applied by EDCDFm are significantly skewed, mean of
  # anomalies might be slightly different from 0. Calculate k correction factor
  # to bring the anomaly mean to 0. This will maintain the GCM projected
  # change in the mean value. This k correction is analogous to the k correction
  # in Pierce et al. Section 3b
  anom_kmean <- 0 - mean(anom_mod_fut_adj)
    
  # Calculate bias-corrected mean value for future time period
  mean_mod_fut_adj <- mean_bc_delta + mean_mod_fut
    
  # Add adjusted future mean and k correction to bias corrected anomalies to
  # get final bias corrected values
  mod_fut_adj <- anom_mod_fut_adj + anom_kmean + mean_mod_fut_adj
      
  return(mod_fut_adj[index(mod_fut_subset)])
}




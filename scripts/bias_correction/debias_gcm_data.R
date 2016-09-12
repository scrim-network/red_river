# Script to bias correct CMIP5 model realizations using Aphrodite observations

Sys.setenv(TZ="UTC")
library(RNetCDF)
library(ncdf4)
library(xts)
library(doParallel)
library(foreach)
source('esd_r/config.R')
source('esd_r/debias.R')

path_in_cmip5 <- cfg['path_cmip5_cleaned']
path_out_cmip5 <- cfg['path_cmip5_debiased']
paths_in = Sys.glob(file.path(path_in_cmip5,"*"))
paths_out = sapply(basename(paths_in),function(x){file.path(path_out_cmip5,x)}, USE.NAMES=FALSE)
for (apath in paths_out) dir.create(apath, showWarnings=FALSE, recursive=TRUE)
fpaths_in <- Sys.glob(file.path(paths_in,'*'))

# Remove files already completed
fpaths_out_exist <- Sys.glob(file.path(paths_out,'*'))
mask_incmplt <- !(basename(fpaths_in) %in% basename(fpaths_out_exist))
fpaths_in <- fpaths_in[mask_incmplt]

ds_obs_tair <- nc_open(file.path(cfg['path_aphrodite_resample'],
                                 'aprhodite_redriver_sat_1961_2007_1deg.nc'))
ds_obs_prcp <- nc_open(file.path(cfg['path_aphrodite_resample'],
                                 'aprhodite_redriver_pcp_1961_2007_1deg.nc'))

a_obs_tair <-  ncvar_get(ds_obs_tair)
a_obs_prcp <-  ncvar_get(ds_obs_prcp)

times_obs <- as.POSIXct(as.character(ds_obs_tair$dim$time$vals),format='%Y%m%d')

nc_close(ds_obs_tair)
nc_close(ds_obs_prcp)

start_yr_base <- substr(cfg['start_date_baseline'],1,4)
end_yr_base <- substr(cfg['end_date_baseline'],1,4)

idx_train <- paste(start_yr_base,end_yr_base,sep='/')
idx_fut <- c(idx_train,'2006/2039','2040/2069','2070/2099')
idx_all <- unique(c(idx_train,idx_fut))
winsize = 31
ds_ptype <- nc_open(fpaths_in[1])
a_ptype <- ncvar_get(ds_ptype)
times_ptype <- utcal.nc(ds_ptype$dim$time$units, ds_ptype$dim$time$vals, type="c")
nc_close(ds_ptype)
mod_ptype <- xts(a_ptype[1,1,],times_ptype)
obs_ptype <- xts(a_obs_tair[1,1,],times_obs)
merge_ptype <- merge.xts(obs_ptype, mod_ptype)
win_masks <- build_window_masks(merge_ptype,idx_all,winsize=winsize)
win_masks1 <- build_window_masks(merge_ptype,idx_all,winsize=1)

bias_funcs <- list('tas'=edqmap,'pr'=edqmap_prcp)
a_obs <- list('tas'=a_obs_tair, 'pr'=a_obs_prcp)

fpath_log <- file.path(cfg['path_logs'], 'bias_correct.log')
writeLines(c(""),fpath_log)
cl <- makeCluster(19)
registerDoParallel(cl)

results <- foreach(fpath=fpaths_in) %dopar% {
  
  library(ncdf4)
  library(xts)
  source(file.path(cfg['code_root'],'esd_r','debias.R'))
  sink(fpath_log, append=TRUE)
  
  cat(sprintf("Started processing %s \n", fpath), file=fpath_log, append=TRUE)
  
  ds <- nc_open(fpath)
  elem <- names(ds$var)
  bias_func <- bias_funcs[[elem]]
  a <- ncvar_get(ds)
  
  a_debias <- array(NA,dim(a))
  
  for (r in 1:dim(a_debias)[1]) {
    
    for (c in 1:dim(a_debias)[2]) {
      
      mod_rc <- xts(a[r,c,],times_ptype)
      obs_rc <- xts(a_obs[[elem]][r,c,],times_obs)
      merge_rc <- merge.xts(obs_rc, mod_rc)
      xtsAttributes(merge_rc) <- list(modname=sprintf('%s_r%dc%d',basename(fpath),r,c))
      a_debias[r,c,] <- mw_bias_correct(merge_rc[,1], merge_rc[,2], idx_train,
                                        idx_fut, bias_func, win_masks, win_masks1)
      
    }
    
  }
  
  # Output debiased data
  
  # Create output dimensions
  dim_lon <- ncdim_def('lon',units='',vals=ds$dim[['lon']]$vals)
  dim_lat <- ncdim_def('lat',units='',vals=ds$dim[['lat']]$vals)
  dim_time <- ncdim_def('time', units=ds$dim[['time']]$units, ds$dim[['time']]$vals,
                        calendar=ds$dim[['time']]$calendar)
  # Create output variable
  var_out <- ncvar_def(elem, units=ds$var[[elem]]$units,
                       dim=list(dim_lon,dim_lat,dim_time), missval=NA, prec='double')
  
  fpath_split <- strsplit(fpath,.Platform$file.sep)[[1]]
  path_rcp <- fpath_split[length(fpath_split)-1]
  fname_out <- fpath_split[length(fpath_split)]
  fpath_out <- file.path(path_out_cmip5,path_rcp,fname_out)
  
  # Create output dataset
  ds_out <- nc_create(fpath_out, var_out, force_v4=TRUE)
  
  # Write data
  ncvar_put(ds_out, vals=a_debias)
  
  # Write global attributes
  attrs_global <- ncatt_get(ds,0)
  for (attname in names(attrs_global)) {
    ncatt_put(ds_out,0,attname,attrs_global[[attname]])
  }
  
  # Close
  nc_close(ds_out)
  nc_close(ds)
  
  cat(sprintf("Finished processing %s \n", fpath), file=fpath_log, append=TRUE)
  
  sink()
  
  return(1)
      
}
stopCluster(cl)

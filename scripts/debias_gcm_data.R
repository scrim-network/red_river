Sys.setenv(TZ="UTC")
library(RNetCDF)
library(ncdf4)
library(xts)

ds <- nc_open('/storage/home/jwo118/group_store/red_river/cmip5_cleaned/historical+rcp26_merged/tas.day.HadGEM2-ES.historical+rcp26.r3i1p1.19760101-20991231.nc')
ds_aphro <- nc_open('/storage/home/jwo118/group_store/red_river/aphrodite/resample/APHRODITE.SAT.0p25deg.remapcon.1deg.nc')

a <- ncvar_get(ds)
a_aphro <- ncvar_get(ds_aphro)

times <- ds$dim$time$vals
times <- utcal.nc(ds$dim$time$units, times, type="c")

times_aphro <- as.POSIXct(as.character(ds_aphro$dim$time$vals),format='%Y%m%d')

mod_ptype <- xts(a[1,1,],times)
obs_ptype <- xts(a_aphro[1,1,],times_aphro)
merge_ptype <- merge.xts(obs_ptype, mod_ptype)

idx_train <- '1976/2005'
idx_fut <- c('1976/2005','2006/2039','2040/2069','2070/2099')
idx_all <- unique(c(idx_train,idx_fut))
winsize = 31

win_masks <- build_window_masks(merge_ptype,idx_all,winsize=winsize)
win_masks1 <- build_window_masks(merge_ptype,idx_all,winsize=1)

a_debias <- array(NA,dim(a))

i <- 0

for (r in 1:dim(a_debias)[1]) {
  
  for (c in 1:dim(a_debias)[2]) {
    
    mod_rc <- xts(a[r,c,],times)
    obs_rc <- xts(a_aphro[r,c,],times_aphro)
    merge_rc <- merge.xts(obs_rc, mod_rc)
    
    a_debias[r,c,] <- mw_bias_correct(obs_rc, mod_rc, idx_train, idx_fut, edqmap, win_masks, win_masks1)
    i = i + 1
    print(i)
  }
  
}


a_debias <- apply(a,c(1,2),function(x){x})
a_debias <- aperm(a_debias,c(2,3,1))

# Create output dimensions
dim_lon <- ncdim_def('lon',units='',vals=ds$dim[['lon']]$vals)
dim_lat <- ncdim_def('lat',units='',vals=ds$dim[['lat']]$vals)
dim_time <- ncdim_def('time', units=ds$dim[['time']]$units,
                      ds$dim[['time']]$vals,
                      calendar=ds$dim[['time']]$calendar)
# Create output variable
var_out <- ncvar_def(ds$var[[1]]$name, units=ds$var[[1]]$units, dim=list(dim_lon,dim_lat,dim_time), missval=NA, prec='double')

# Create output dataset
ds_out <- nc_create('/storage/home/jwo118/temp/test.nc',var_out,force_v4=TRUE)

# Write data
ncvar_put(ds_out,vals=a_debias)

# Close
nc_close(ds_out)



#library(sp)
#library(doParallel)
#library(foreach)
#library(rgdal)
#library(gstat)
#library(spacetime)
#library(xts)
#
#elem <- commandArgs(TRUE)[1]
#print(paste0("Running for ",elem))
#
#ushcn_anom <- read_anoms(cfg$USHCN_CONFIG[[paste0('FPATH_ANN_ANOMS_',toupper(elem))]])
#
## Load previously calculated variogram
#load(file.path(cfg$USHCN_CONFIG$PATH_RDATA, 'ushcn_vgms.RData'))        
#elem_vgm <- get(paste0('vgm_', elem))
#
## Load interpolation raster grid
#agrid <- raster(file.path(cfg$USHCN_CONFIG$PATH_RASTER, 'mask_conus_25deg.tif'))
## Make sure projection is the same as stns.
#proj4string(agrid) <- proj4string(ushcn_anom)
## Convert to SpatialPixelsDataFrame to define grid
## for interpolations
#spdf_agrid <- as(agrid,'SpatialPixelsDataFrame')
#
## Get years over which to interpolate
#uyrs <- as.character(xts::.indexyear(ushcn_anom@time) + 1900)
#
#fpath_log <- file.path(cfg$USHCN_CONFIG$PATH_LOG, 'interp_anom_by_year.log')
#writeLines(c(""),fpath_log)
#cl <- makeCluster(3)
#registerDoParallel(cl)
#
#
#anom_stack <- foreach(yr=uyrs) %dopar%
#    {
#      library(spacetime)
#      library(gstat)
#      library(raster)
#      
#      anom_kriged <- krige(as.formula(paste0(elem,'~1')), ushcn_anom[, yr, elem],
#          spdf_agrid, model=elem_vgm, nmax=100)
#      
#      anom_kriged <- as(anom_kriged['var1.pred'],'RasterLayer')
#      
#      fpath_out <- file.path(cfg$USHCN_CONFIG$PATH_RASTER_ANN_ANOMS,
#          paste0("ann_anom_",elem,"_",yr,".tif"))
#      writeRaster(anom_kriged,fpath_out)
#      cat(sprintf("Finished processing %s \n",yr), file=fpath_log, append=TRUE)
#      
#      return(anom_kriged)
#      
#    }
#stopCluster(cl)

anom_stack <- stack(anom_stack)

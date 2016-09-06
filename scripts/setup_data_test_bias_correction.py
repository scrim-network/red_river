import xarray as xr
import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    ds = xr.open_dataset('/storage/home/jwo118/group_store/red_river/aphrodite/resample/APHRODITE.SAT.0p25deg.remapcon.1deg.nc')
    
    # Convert times to datetime format
    times = pd.to_datetime(ds.time.to_pandas().astype(np.str),
                           format='%Y%m%d', errors='coerce')        
    ds['time'] = times.values
    
    ann_sat = ds.SAT[:,2,3].resample('MS', dim='time', how='mean').resample('AS',dim='time',how='mean')
    
    
    climatology = ds.groupby('time.month').mean('time')
    
    ds_ccsm = xr.open_dataset('/storage/home/jwo118/group_store/red_river/cmip5_resample/historical+rcp85_merged/tas.day.CCSM4.historical+rcp85.r6i1p1.19500101-21001231.nc')
    # Convert times to datetime format
    times = pd.to_datetime(ds_ccsm.time.to_pandas().astype(np.str),
                           format='%Y%m%d.5', errors='coerce')        
    ds_ccsm['time'] = times.values
    
    k_to_c = lambda k: k - 273.15
    
    ann_tas = k_to_c(ds_ccsm.tas[:,2,3].loc['1961-01-01':'2007-12-31'].resample('MS', dim='time', how='mean').resample('AS',dim='time',how='mean'))
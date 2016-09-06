import xarray as xr
import pandas as pd
import numpy as np

def unit_convert_tas(da):
    
    assert da.units == 'K'
    return ds.tas-273.15

if __name__ == '__main__':

    # 360 day calendar
    ds = xr.open_dataset('/storage/home/jwo118/group_store/red_river/cmip5_resample/historical+rcp60_merged/pr.day.HadGEM2-AO.historical+rcp60.r1i1p1.18600101-21001230.nc')

    # 365 day calendar
    ds = xr.open_dataset('/storage/home/jwo118/group_store/red_river/cmip5_resample/historical+rcp85_merged/pr.day.CCSM4.historical+rcp85.r6i1p1.19500101-21001231.nc')
    ds = xr.open_dataset('/storage/home/jwo118/group_store/red_river/cmip5_resample/historical+rcp45_merged/pr.day.GFDL-CM3.historical+rcp45.r3i1p1.18600101-21001231.nc')

    # Get rid of days > 2101 due to pandas Timestamp range limitation
    ds = ds.sel(time=ds.time[ds.time <= 21010101])
    
    # Convert times to datetime format
    times = pd.to_datetime(ds.time.to_pandas().astype(np.str),
                           format='%Y%m%d.5', errors='coerce')        
    ds['time'] = times.values
    # Get rid of invalid dates (occurs with 360-day calendar)
    ds = ds.isel(time=np.nonzero(ds.time.notnull().values)[0])
    
    # Full sequence of valid dates
    # Todo: get 12-31
    times_full = pd.date_range(ds.time[0].values, ds.time[-1].values)
    
    # Reindex to full sequence and place NA for missing values
    #da = ds.tas.reindex(time=times_full)
    da = ds.pr.reindex(time=times_full)
    
    s = da.to_series() #index: time, lat, lon
    s = s.reorder_levels(['lat','lon','time'])
    s = s.sortlevel(0, sort_remaining=True)
    s = s.unstack(['lat','lon'])
    s = s.interpolate(method='time')
    new_da = xr.DataArray.from_series(s.stack(['lat','lon']))
    #new_da = new_da-273.15 #convert K to C
    new_da = new_da*86400 #convert kg m-2 s-1 to mm day-1 (86400 = # of sec in day)
    
    #ds_obs = xr.open_dataset('/storage/home/jwo118/group_store/red_river/aphrodite/resample/APHRODITE.SAT.0p25deg.remapcon.1deg.nc')
    ds_obs = xr.open_dataset('/storage/home/jwo118/group_store/red_river/aphrodite/resample/APHRODITE.PCP.0p25deg.remapcon.1deg.nc')
    #obs_da = ds_obs.SAT
    obs_da = ds_obs.PCP
    # Convert times to datetime format
    times = pd.to_datetime(obs_da.time.to_pandas().astype(np.str),
                           format='%Y%m%d', errors='coerce')        
    obs_da['time'] = times.values
        
    df = pd.DataFrame({'obs':obs_da.loc['1976':'2005',:,:][:,3,3].to_pandas(),
                       'model':new_da.loc['1976':,:,:][:,3,3].to_pandas()})
    
    # Plot monthly normals
    
    #Prcp
   
    # Ann prcp plot
    mthly = df.loc['1976':'2005'].resample('MS').sum()
    mthly.resample('AS').sum().plot()
    
    # Monthly normals plot 
    mthly.groupby(mthly.index.month).mean().plot()
    
    # Monthly normals # of prcp days plot
    wet = df.loc['1976':'2005'] > 0
    mthly_wet = wet.resample('MS').sum()
    mthly_wet.groupby(mthly_wet.index.month).mean().plot()
    
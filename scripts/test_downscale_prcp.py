import xarray as xr
import pandas as pd
import numpy as np

def convert_times(da):
    
    times = pd.to_datetime(da.time.to_pandas().astype(np.str),
                           format='%Y%m%d', errors='coerce')        
    return times.values

def window_masks(time_idx, winsize=91):
    
    delta = pd.Timedelta(days=np.round((float(winsize)-1)/2))

    mth_day = time_idx.strftime('%m-%d')
  
    dates_yr = pd.date_range('2007-01-01', '2007-12-31')
    
    win_masks = pd.DataFrame(np.ones((dates_yr.size,time_idx.size), dtype=np.bool),
                             index=dates_yr.strftime('%m-%d'),columns=time_idx,dtype=np.bool)
    
    for i,a_date in enumerate(dates_yr):
        
        mth_day_win = pd.date_range(a_date - delta, a_date + delta).strftime('%m-%d')
        win_masks.values[i,:] = np.in1d(mth_day, mth_day_win)
    

   
    # Add month day win for leap day
    a_date = pd.Timestamp('2008-02-29')
    mth_day_win = pd.date_range(a_date - delta, a_date + delta).strftime('%m-%d')
    
    # Direct assignment via loc/iloc very slow?
    win_masks.loc['02-29',:] = True
    win_masks.values[-1,:] = np.in1d(mth_day, mth_day_win)
    
    win_masks = win_masks.sort_index(axis=0)
    win_masks = win_masks.astype(np.bool)
    
    return win_masks

if __name__ == '__main__':
    
    start_year_train = '1961'
    end_year_train = '1990'
    start_year_test = '1991'
    end_year_test = '2007'
    
    start_year_train = '1978'
    end_year_train = '2007'
    start_year_test = '1961'
    end_year_test = '1977'
    
    
    da_aphro = xr.open_dataset('/storage/home/jwo118/group_store/red_river/aphrodite/APHRODITE.Monsoon_Asia.PCP.0p25deg.v1101r2.1961-2007.nc').PCP
    da_aphro_cg = xr.open_dataset('/storage/home/jwo118/group_store/red_river/aphrodite/resample/APHRODITE.PCP.0p25deg.remapcon.1deg.buf.remapbic.p25deg.nc').PCP
    
    da_aphro['time'] = convert_times(da_aphro)
    da_aphro_cg['time'] = convert_times(da_aphro_cg)
    
    # Subset original aphrodite dataset to Red River Domain
    da_aphro = da_aphro.loc[:,da_aphro_cg.lat.values,da_aphro_cg.lon.values]
    da_aprho = da_aphro.load()
    
    # Use validation time period as the test "GCM"
    da_gcm = da_aphro_cg.loc[start_year_test:end_year_test].load()
    # DataArray to store downscaled results
    da_gcm_d = da_gcm.copy()
    
    # Subset da_aphro_cg to training period and load
    da_aphro_cg = da_aphro_cg.loc[start_year_train:end_year_train].load()
    win_masks = window_masks(da_aphro_cg.time.to_pandas().index)
    
    for a_date in da_gcm.time.to_pandas().index:
        
        print a_date
        
        vals_gcm = da_gcm.loc[a_date]
        analog_pool = da_aphro_cg[win_masks.loc[a_date.strftime('%m-%d')].values]
        rmse_analogs = np.sqrt((np.square(vals_gcm-analog_pool)).mean(dim=('lon','lat')))
        vals_analog = analog_pool[rmse_analogs.values.argmin()]
    
        s = vals_gcm/vals_analog
        s.values[np.logical_or(np.isinf(s.values),np.isnan(s.values))] = 0
        s.values[vals_analog.isnull().values] = np.nan
        
        # Limit scaling factor by 6 standard deviations
        try:
             
            st = s.copy()
            st.values[st.values==0] = np.nan
             
            # Limit s by 6 std
            sz = ((st - st.mean())/st.std())
            max_s = s.values[sz.values <= 6].max()
            min_s = s.values[sz.values >= -6].min()
            s.values[sz.values > 6] = max_s
            s.values[sz.values < -6] = min_s
        except ValueError:
            pass
                
        vals_d = da_aphro.loc[vals_analog.time.values]*s
        
        da_gcm_d.loc[a_date] = vals_d
        
    ann_mean_d = da_gcm_d.resample('AS', dim='time', how='sum',skipna=False).mean(dim='time')
    ann_mean_obs = da_aphro.loc[start_year_test:end_year_test].resample('AS', dim='time', how='sum',skipna=False).mean(dim='time')

    (((ann_mean_d-ann_mean_obs)/ann_mean_obs)*100).plot()
    
    nwet_d = (da_gcm_d > 0).sum(dim='time')
    nwet_obs = (da_aphro.loc[start_year_test:end_year_test] > 0).sum(dim='time').astype(np.float)
    
    

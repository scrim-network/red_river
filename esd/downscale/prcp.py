'''
Functions for downscaling precipitation.
'''

import numpy as np
import pandas as pd
import xarray as xr

def _convert_times(da):
    
    times = pd.to_datetime(da.time.to_pandas().astype(np.str),
                           format='%Y%m%d', errors='coerce')        
    return times.values

def _window_masks(time_idx, winsize=91):
    
    delta = pd.Timedelta(days=np.round((float(winsize) - 1) / 2))

    mth_day = time_idx.strftime('%m-%d')
  
    dates_yr = pd.date_range('2007-01-01', '2007-12-31')
    
    win_masks = pd.DataFrame(np.ones((dates_yr.size, time_idx.size), dtype=np.bool),
                             index=dates_yr.strftime('%m-%d'), columns=time_idx, dtype=np.bool)
    
    for i, a_date in enumerate(dates_yr):
        
        mth_day_win = pd.date_range(a_date - delta, a_date + delta).strftime('%m-%d')
        win_masks.values[i, :] = np.in1d(mth_day, mth_day_win)
    
    # Add month day win for leap day
    a_date = pd.Timestamp('2008-02-29')
    mth_day_win = pd.date_range(a_date - delta, a_date + delta).strftime('%m-%d')
    
    # Direct assignment via loc/iloc very slow?
    win_masks.loc['02-29', :] = True
    win_masks.values[-1, :] = np.in1d(mth_day, mth_day_win)
    
    win_masks = win_masks.sort_index(axis=0)
    win_masks = win_masks.astype(np.bool)
    
    return win_masks

def setup_data_for_analog(fpath_prcp_obs, fpath_prcp_obsc, fpath_prcp_mod,
                          base_start_year, base_end_year, downscale_start_year,
                          downscale_end_year):
    
    da_obs = xr.open_dataset(fpath_prcp_obs).PCP
    da_obsc = xr.open_dataset(fpath_prcp_obsc).PCP
    da_mod = xr.open_dataset(fpath_prcp_mod).PCP

    da_obs['time'] = _convert_times(da_obs)
    da_obsc['time'] = _convert_times(da_obsc)
    da_mod['time'] = _convert_times(da_mod)

    da_obs = da_obs.loc[:, da_mod.lat.values, da_mod.lon.values]
    da_obs = da_obs.load()

    da_obs_mthly = da_obs.resample('MS', dim='time', how='mean', skipna=False)
    da_obsc_mthly = da_obsc.resample('MS', dim='time', how='mean', skipna=False)
    da_mod_mthly = da_mod.resample('MS', dim='time', how='mean', skipna=False)
    da_obs_clim = da_obs_mthly.loc[base_start_year:base_end_year].groupby('time.month').mean(dim='time')
    da_obsc_clim = da_obsc_mthly.loc[base_start_year:base_end_year].groupby('time.month').mean(dim='time')
    da_mod_clim = da_mod_mthly.loc[base_start_year:base_end_year].groupby('time.month').mean(dim='time')
    
    da_obs_anoms = da_obs.groupby('time.month') / da_obs_clim
    da_obsc_anoms = da_obsc.groupby('time.month') / da_obsc_clim
    da_mod_anoms = da_mod.groupby('time.month') / da_mod_clim
    
    da_obs_anoms = da_obs_anoms.drop('month')
    da_obsc_anoms = da_obsc_anoms.drop('month')
    da_mod_anoms = da_mod_anoms.drop('month')
    
    da_mod = da_mod.loc[downscale_start_year:downscale_end_year]
    da_mod_anoms = da_mod_anoms.loc[downscale_start_year:downscale_end_year]
    
    da_obsc_anoms = da_obsc_anoms.loc[base_start_year:base_end_year]
    da_obsc = da_obsc.loc[base_start_year:base_end_year]
    win_masks = _window_masks(da_obsc_anoms.time.to_pandas().index)
    
    da_obs.name = 'obs'
    da_obs_anoms.name = 'obs_anoms'
    da_obs_clim.name = 'obs_clim'
    da_obsc.name = 'obsc'
    da_obsc_anoms.name = 'obsc_anoms'
    da_mod.name = 'mod'
    da_mod_anoms.name = 'mod_anoms'
    
    ds = xr.merge([da_obs, da_obs_anoms, da_obs_clim, da_obsc, da_obsc_anoms, da_mod, da_mod_anoms])
    
    ds.attrs['base_start_year'] = base_start_year
    ds.attrs['base_end_year'] = base_end_year
    ds.attrs['downscale_start_year'] = downscale_start_year
    ds.attrs['downscale_end_year'] = downscale_end_year
    
    return ds, win_masks

def downscale_analog_anoms(ds, win_masks):
    
    da_mod_anoms = ds.mod_anoms.loc[ds.downscale_start_year:ds.downscale_end_year]
    da_obsc_anoms = ds.obsc_anoms.loc[ds.base_start_year:ds.base_end_year]
    da_obs_anoms = ds.obs_anoms
    da_obs_clim = ds.obs_clim
    
    # DataArray to store downscaled results
    da_mod_anoms_d = da_mod_anoms.copy()
    
    for a_date in da_mod_anoms.time.to_pandas().index:
        
        print a_date
        
        vals_mod = da_mod_anoms.loc[a_date]
        analog_pool = da_obsc_anoms[win_masks.loc[a_date.strftime('%m-%d')].values]
        # mae_analogs = np.abs(vals_gcm-analog_pool).mean(dim=('lon','lat'))
        # mae_wet_day_analogs = np.abs((vals_gcm>0)-(analog_pool>0)).mean(dim=('lon','lat'))
        # vals_analog = analog_pool[int((mae_wet_day_analogs/mae_wet_day_analogs.mean() + mae_analogs/mae_analogs.mean()).argmin())]
        # vals_analog = analog_pool[int(mae_analogs.argmin())]
        
        rmse_analogs = np.sqrt((np.square(vals_mod - analog_pool)).mean(dim=('lon', 'lat')))
        vals_analog = analog_pool[int(rmse_analogs.argmin())]
        
        s = vals_mod - vals_analog
        vals_obs_anoms = da_obs_anoms.loc[vals_analog.time.values]
        vals_d = vals_obs_anoms.copy()
        mask_wet = vals_d.values > 0        
        vals_d.values[mask_wet] = vals_d.values[mask_wet] + s.values[mask_wet]
        
        mask_wet_invalid = np.logical_and(mask_wet, vals_d.values <= 0)        
        vals_d.values[mask_wet_invalid] = vals_obs_anoms.values[mask_wet_invalid]
                   
        da_mod_anoms_d.loc[a_date] = vals_d.values
    
    da_mod_d = da_mod_anoms_d.groupby('time.month') * da_obs_clim
    
    return da_mod_d
    
def downscale_analog(ds, win_masks):
        
    da_mod = ds.mod.loc[ds.downscale_start_year:ds.downscale_end_year]    
    da_obsc = ds.obsc.loc[ds.base_start_year:ds.base_end_year]
    da_obs = ds.obs
    
    da_mod_d = da_mod.copy()
    
    for a_date in da_mod.time.to_pandas().index:
        
        print a_date
        
        vals_mod = da_mod.loc[a_date]
        analog_pool = da_obsc[win_masks.loc[a_date.strftime('%m-%d')].values]
        rmse_analogs = np.sqrt((np.square(vals_mod - analog_pool)).mean(dim=('lon', 'lat')))
        vals_analog = analog_pool[int(rmse_analogs.argmin())]
            
        s = vals_mod / vals_analog
        s.values[np.logical_or(np.isinf(s.values), np.isnan(s.values))] = 0
        s.values[vals_analog.isnull().values] = np.nan
        
        # Limit scaling factor by 6 standard deviations
        try:
            st = s.copy()
            st.values[st.values == 0] = np.nan
             
            # Limit s by 6 std
            sz = ((st - st.mean()) / st.std())
            max_s = s.values[sz.values <= 6].max()
            min_s = s.values[sz.values >= -6].min()
            s.values[sz.values > 6] = max_s
            s.values[sz.values < -6] = min_s
        except ValueError:
            pass
                
        vals_d = da_obs.loc[vals_analog.time.values] * s
        
        da_mod_d.loc[a_date] = vals_d
    
    return da_mod_d

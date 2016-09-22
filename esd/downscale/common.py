'''
Common downscaling utility functions.
'''

from esd.util import StatusCheck, grt_circle_dist
import itertools
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

def validate_correlation(mod_d, obs, rfunc, months=None):
        
    if months is not None:
        
        mask_season = mod_d['time.month'].to_pandas().isin(months).values
        mod_d = mod_d[mask_season,:,:]
        obs = obs[mask_season,:,:]
    
    cors = np.empty(mod_d[0].shape)
    cors.fill(np.nan)
    rows,cols = np.nonzero((~mod_d[0].isnull()).values)
    
    for r,c, in zip(rows,cols):
        cors[r,c] = rfunc(mod_d[:,r,c].values, obs[:,r,c].values)
        
    cors = xr.DataArray(cors,coords=[mod_d.lat,mod_d.lon],name='r')
            
    return cors

def validate_spatial_sd_monthly(da):
        
    mthly_sd = da.std(dim=['lat','lon'], ddof=1).groupby('time.month').mean()
    return mthly_sd.to_pandas()

def validate_quantile(mod_d, obs, q, months=None):
    
    q_func = lambda a,axis: np.nanpercentile(a,q,axis) 
    
    if months is not None:
        
        mask_season = mod_d['time.month'].to_pandas().isin(months).values
        mod_d = mod_d[mask_season,:,:]
        obs = obs[mask_season,:,:]
        
    mod_d_q = mod_d.reduce(q_func, dim='time')
    obs_q = obs.reduce(q_func, dim='time')
                
    # Calculate percent and absolute error
    err = mod_d_q-obs_q
    pct_err = (err/obs_q)*100
    obs_q.name = 'obs'
    mod_d_q.name = 'mod'
    err.name = 'err'
    pct_err.name = 'pct_err'
    
    return xr.merge([err, pct_err, obs_q, mod_d_q])

def validate_spatial_coherence(da_prcp, rfunc, months=None):
    
    if months is not None:
        
        mask_season = da_prcp['time.month'].to_pandas().isin(months).values
        da_prcp = da_prcp[mask_season,:,:]
    
    s = da_prcp.to_series()
    s = s.reorder_levels(['lat','lon','time'])
    s = s.sortlevel(0, sort_remaining=True)
    s = s.unstack(['lat','lon'])
    
    gridcell_i = np.arange(s.shape[1])
    gridcell_pairs = itertools.combinations(gridcell_i, 2)
    gridcell_pairs = [p for p in gridcell_pairs]
    
    cors = np.empty(len(gridcell_pairs))
    dist = np.empty(len(gridcell_pairs))
    
    schk = StatusCheck(cors.size, 2000)
    for x,a_pair in enumerate(gridcell_pairs):
        
        a_pair = list(a_pair)
        
        pair_obs = s.iloc[:, a_pair]
        rho = rfunc(pair_obs.iloc[:, 0], pair_obs.iloc[:, 1])
        cors[x] = rho
        
        lat1,lon1 =  s.columns[a_pair[0]]
        lat2,lon2 =  s.columns[a_pair[1]]
        
        dist[x] = grt_circle_dist(lon1, lat1, lon2, lat2)
        
        schk.increment()
    
    df_cld = pd.DataFrame({'r':cors, 'dist':dist})
    
    def calc_lag_means(df, nbins):
    
        df['lag'] = pd.cut(df.dist, bins=nbins)
        df_means = df.groupby('lag').mean()
        df_means['rinv'] = 1.0 - df_means['r']
        return df_means
    
    
    dfm = calc_lag_means(df_cld, nbins=40).set_index('dist')
    dfm.index.name = 'Spatial Lag (km)'

    return dfm

def validate_temporal_sd(mod_d, obs, months=None):
        
    if months is not None:
        
        mask_season = mod_d['time.month'].to_pandas().isin(months).values
        mod_d = mod_d[mask_season,:,:]
        obs = obs[mask_season,:,:]
    
    mod_d_sd = mod_d.std(dim='time', skipna=False, ddof=1)
    obs_sd = obs.std(dim='time', skipna=False, ddof=1)
    # Calculate percent and absolute error
    err = mod_d_sd-obs_sd
    pct_err = (err/obs_sd)*100
    obs_sd.name = 'obs'
    mod_d_sd.name = 'mod'
    err.name = 'err'
    pct_err.name = 'pct_err'
    
    return xr.merge([err, pct_err, obs_sd, mod_d_sd]) 

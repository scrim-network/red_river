'''
Functions for downscaling precipitation and validation of the downscaled results.
'''

from .common import _convert_times, _window_masks
from esd.util.misc import runs_of_ones_array
from scipy.interpolate.ndgriddata import griddata
from scipy.spatial.qhull import QhullError
import numpy as np
import statsmodels.api as sm
import xarray as xr

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


def scale_anoms(s, obs_anoms):
    
    vals_d = obs_anoms.copy()
    mask_wet = vals_d.values > 0        
    vals_d.values[mask_wet] = vals_d.values[mask_wet] + s.values[mask_wet]
    mask_wet_invalid = np.logical_and(mask_wet, vals_d.values <= 0)        
    
    has_invalid = mask_wet_invalid.any()
    
    try:
    
        if has_invalid:
        
            s.values[np.logical_or(mask_wet_invalid, ~mask_wet)] = np.nan
            
            r,c = np.nonzero(np.isfinite(s.values))
            y = s.lat[r].values
            x = s.lon[c].values
            points = (x,y)
            values = s.values[r,c]
            grid_x,grid_y = np.meshgrid(s.lon.values, s.lat.values)
            
            try:
                sgrid = griddata(points, values, (grid_x,grid_y), method='cubic')
            except QhullError:
                sgrid = np.empty(s.shape)
                sgrid.fill(np.nan)
            
            sgrid_n = griddata(points, values, (grid_x,grid_y), method='nearest')
                    
            mask_nan = np.isnan(sgrid)
            sgrid[mask_nan] = sgrid_n[mask_nan]
            
            s = xr.DataArray(sgrid, coords=[s.lat,s.lon])
    
            vals_d = obs_anoms.copy()
            mask_wet = vals_d.values > 0        
            vals_d.values[mask_wet] = vals_d.values[mask_wet] + s.values[mask_wet]
            mask_wet_invalid = np.logical_and(mask_wet, vals_d.values <= 0)        
            
            has_invalid = mask_wet_invalid.any()
            
            if has_invalid:
                
                vals_d.values[mask_wet_invalid] = np.nan
                
                r,c = np.nonzero(vals_d.values > 0)
                y = vals_d.lat[r].values
                x = vals_d.lon[c].values
                points = (x,y)
                values = vals_d.values[r,c]
                grid_x,grid_y = np.meshgrid(vals_d.lon.values, vals_d.lat.values)
                
                try:
                    vals_d_grid = griddata(points, values, (grid_x,grid_y), method='cubic')
                except QhullError:
                    vals_d_grid = np.zeros(vals_d.shape)
                
                try:
                    vals_d_grid_l = griddata(points, values, (grid_x,grid_y), method='linear')
                except QhullError:
                    vals_d_grid_l = np.empty(vals_d.shape)
                    vals_d_grid_l.fill(np.nan)
                
                vals_d_grid_n = griddata(points, values, (grid_x,grid_y), method='nearest')
                
                mask_zero  = vals_d_grid <= 0
                vals_d_grid[mask_zero] = vals_d_grid_l[mask_zero]
                
                mask_nan = np.isnan(vals_d_grid)
                vals_d_grid[mask_nan] = vals_d_grid_n[mask_nan]
                
                vals_d = xr.DataArray(vals_d_grid, coords=[vals_d.lat,vals_d.lon])
                vals_d.values[obs_anoms.values == 0] = 0
    
    except ValueError as e:
        
        if e.args[0] == 'No points given':
            # No valid scaler factors
            vals_d = obs_anoms.copy()
        else:
            raise
        
    return vals_d
    

def downscale_analog_anoms_sgrid(ds, win_masks):
    
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
        rmse_analogs = np.sqrt((np.square(vals_mod - analog_pool)).mean(dim=('lon', 'lat')))
        vals_analog = analog_pool[int(rmse_analogs.argmin())]
        
        s = vals_mod - vals_analog
        vals_obs_anoms = da_obs_anoms.loc[vals_analog.time.values]
        vals_d = scale_anoms(s, vals_obs_anoms)
                           
        da_mod_anoms_d.loc[a_date] = vals_d.values
    
    da_mod_d = da_mod_anoms_d.groupby('time.month') * da_obs_clim
    
    return da_mod_d


def downscale_analog_anoms_noscale(ds, win_masks):
    
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
        rmse_analogs = np.sqrt((np.square(vals_mod - analog_pool)).mean(dim=('lon', 'lat')))
        vals_analog = analog_pool[int(rmse_analogs.argmin())]
        
        vals_obs_anoms = da_obs_anoms.loc[vals_analog.time.values]                  
        da_mod_anoms_d.loc[a_date] = vals_obs_anoms.values
    
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

def _setnull_s_outliers(s):
    
    try:
         
        # Limit s by 6 std
        sz = ((s - s.mean()) / s.std())
        mask_null = np.logical_or(sz.values > 6, sz.values < -6)
        s.values[mask_null] = np.nan
    except ValueError:
        pass
    
    return s,mask_null.any()
    
    
def process_s(s):
    
    s.values[s.values == 0] = np.nan
    s.values[np.logical_or(np.isinf(s.values), np.isnan(s.values))] = np.nan
    s,had_outlier = _setnull_s_outliers(s)
    while had_outlier:
        s,had_outlier = _setnull_s_outliers(s)
    
    has_null = s.isnull().any()
    
    if has_null:
    
        try:
        
            r,c = np.nonzero(np.isfinite(s.values))
            y = s.lat[r].values
            x = s.lon[c].values
            points = (x,y)
            values = s.values[r,c]
            grid_x,grid_y = np.meshgrid(s.lon.values, s.lat.values)
            
            try:
                sgrid = griddata(points, values, (grid_x,grid_y), method='cubic')
            except QhullError:
                sgrid = np.zeros(s.shape)
            
            try:
                sgrid_l = griddata(points, values, (grid_x,grid_y), method='linear')
            except QhullError:
                sgrid_l = np.empty(s.shape)
                sgrid_l.fill(np.nan)
                
            sgrid_n = griddata(points, values, (grid_x,grid_y), method='nearest')
            
            mask_zero  = sgrid <= 0
            sgrid[mask_zero] = sgrid_l[mask_zero]
            
            mask_nan = np.isnan(sgrid)
            sgrid[mask_nan] = sgrid_n[mask_nan]
            
            s = xr.DataArray(sgrid, coords=[s.lat,s.lon])
        
        except ValueError as e:
            
            if e.args[0] == 'No points given':
                # No valid scaler factors
                s.values[:] = 1
            else:
                raise
    
    return s
    
def downscale_analog_nonzero_scale(ds, win_masks):
        
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
        s = process_s(s)
                
        vals_obs =  da_obs.loc[vals_analog.time.values]
        vals_d = vals_obs.copy()
        mask_wet = vals_d.values > 0 
        vals_d.values[mask_wet] = vals_d.values[mask_wet] * s.values[mask_wet]
        
        da_mod_d.loc[a_date] = vals_d
    
    return da_mod_d

def downscale_analog_noscale(ds, win_masks):
        
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
                            
        vals_d = da_obs.loc[vals_analog.time.values]
        
        da_mod_d.loc[a_date] = vals_d
    
    return da_mod_d

def downscale_linear_combo(da_gcm, da_aphro_cg, win_masks, da_aphro):
    
    da_gcm_d = da_gcm.copy()
    
    for a_date in da_gcm.time.to_pandas().index:
        
        print a_date
        
        vals_gcm = da_gcm.loc[a_date]
        analog_pool = da_aphro_cg[win_masks.loc[a_date.strftime('%m-%d')].values]
        #rmse_analogs = np.sqrt((np.square(vals_gcm-analog_pool)).mean(dim=('lon','lat')))
        mae_analogs = np.abs(vals_gcm-analog_pool).mean(dim=('lon','lat'))
        mae_wet_day_analogs = np.abs((vals_gcm>0).astype(np.float)-(analog_pool>0).astype(np.float)).mean(dim=('lon','lat'))
        
        mae_combo = mae_wet_day_analogs/mae_wet_day_analogs.mean() + mae_analogs/mae_analogs.mean()
        
        analog_pool = analog_pool[mae_combo.argsort()[0:30].values]
        
        #X = analog_pool.values.reshape((30, np.prod(analog_pool.shape[1:]))).T
        X = analog_pool.to_dataframe().unstack('time').values
        # Add column for intercept
        X = sm.add_constant(X)
        
        y = vals_gcm.to_series().values
        
        try:
        
            mask_wet = y > 0
            X = X[mask_wet,:]
            y = y[mask_wet]
                    
            model = sm.OLS(y, X, missing='drop')
            result = model.fit()
            
            analog_coef = xr.DataArray(result.params[1:], coords=[analog_pool.time])
            analog_intercept = result.params[0]
        
        except ValueError:
            analog_intercept = 0
            analog_coef = 0
            
        d_predict = analog_intercept + ((analog_coef * da_aphro.loc[analog_pool.time.values])).sum(dim='time',skipna=False)
        
        mask_dry =  np.logical_or(d_predict.values <= 0,
                                  (da_aphro.loc[analog_pool.time.values[0]] == 0).values)
        
        d_predict.values[mask_dry] = 0
        
        da_gcm_d.loc[a_date] = d_predict.values
    
    return da_gcm_d

def validate_avg_total_prcp(mod_d, obs, months=None):
    
    if months is None:
        
        # Calculate annual avg totals
        mod_d_avg = mod_d.resample('AS', dim='time', how='sum',skipna=False).mean(dim='time')
        obs_avg = obs.resample('AS', dim='time',how='sum', skipna=False).mean(dim='time')
        
    else:
        
        mod_d_mthly = mod_d.resample('1MS', dim='time', how='sum', skipna=False)
        obs_mthly = obs.resample('1MS', dim='time', how='sum', skipna=False)
        
        mask_season = np.in1d(mod_d_mthly['time.month'].values, months)
        mask_season = xr.DataArray(mask_season, coords=[mod_d_mthly.time])
        
        mod_d_mthly = mod_d_mthly.where(mask_season)
        obs_mthly = obs_mthly.where(mask_season)
        
        nmths = len(months)
        mod_d_roller = mod_d_mthly.rolling(min_periods=nmths, center=False, time=nmths)
        obs_roller = obs_mthly.rolling(min_periods=nmths, center=False, time=nmths)
        
        mod_d_avg = mod_d_roller.sum(skipna=False).mean(dim='time')
        obs_avg = obs_roller.sum(skipna=False).mean(dim='time')
        
    # Calculate percent and absolute error
    err = mod_d_avg-obs_avg
    pct_err = (err/obs_avg)*100
    obs_avg.name = 'obs_avg'
    mod_d_avg.name = 'mod_avg'
    err.name = 'err'
    pct_err.name = 'pct_err'
    
    return xr.merge([err, pct_err, obs_avg, mod_d_avg])

def validate_avg_wet_day_frac(mod_d, obs, wet_thres=0.1, months=None):
    
    wet_mod_d = (mod_d > wet_thres).astype(np.float)
    wet_obs = (obs > wet_thres).astype(np.float)
    
    wet_mod_d.values[mod_d.isnull().values] = np.nan
    wet_obs.values[obs.isnull().values] = np.nan
    
    if months is None:
        
        # Calculate annual avg wet fraction
        nwet_mod_d = wet_mod_d.resample('AS', dim='time', how='sum',skipna=False)
        nwet_obs = wet_obs.resample('AS', dim='time', how='sum',skipna=False)
        
        fracwet_mod_d = (nwet_mod_d.groupby('time.year')/wet_mod_d.groupby('time.year').count(dim='time').astype(np.float)).mean(dim='time')
        fracwet_obs = (nwet_obs.groupby('time.year')/wet_obs.groupby('time.year').count(dim='time').astype(np.float)).mean(dim='time')
                
    else:
                
        nwet_mod_d_mthly = wet_mod_d.resample('1MS', dim='time', how='sum', skipna=False)
        nwet_obs_mthly = wet_obs.resample('1MS', dim='time', how='sum', skipna=False)
        
        cnt_mod_d_mthly = wet_mod_d.resample('1MS', dim='time', how=lambda x,axis: np.isfinite(x).sum(axis=axis))
        cnt_obs_mthly = wet_obs.resample('1MS', dim='time', how=lambda x,axis: np.isfinite(x).sum(axis=axis))
        
        mask_season = np.in1d(nwet_mod_d_mthly['time.month'].values, months)
        mask_season = xr.DataArray(mask_season, coords=[nwet_mod_d_mthly.time])
        
        nwet_mod_d_mthly = nwet_mod_d_mthly.where(mask_season)
        nwet_obs_mthly = nwet_obs_mthly.where(mask_season)
        cnt_mod_d_mthly = cnt_mod_d_mthly.where(mask_season)
        cnt_obs_mthly = cnt_obs_mthly.where(mask_season)
        
        nmths = len(months)
        nwet_mod_d_roller = nwet_mod_d_mthly.rolling(min_periods=nmths, center=False, time=nmths)
        nwet_obs_roller = nwet_obs_mthly.rolling(min_periods=nmths, center=False, time=nmths)
        cnt_mod_d_roller = cnt_mod_d_mthly.rolling(min_periods=nmths, center=False, time=nmths)
        cnt_obs_roller = cnt_obs_mthly.rolling(min_periods=nmths, center=False, time=nmths)
        
        fracwet_mod_d = (nwet_mod_d_roller.sum(skipna=False)/cnt_mod_d_roller.sum(skipna=False)).mean(dim='time')
        fracwet_obs = (nwet_obs_roller.sum(skipna=False)/cnt_obs_roller.sum(skipna=False)).mean(dim='time')
        
    # Calculate percent and absolute error
    err = fracwet_mod_d-fracwet_obs
    fracwet_obs.name = 'obs_avg'
    fracwet_mod_d.name = 'mod_avg'
    err.name = 'err'
    
    return xr.merge([err*100, fracwet_obs*100, fracwet_mod_d*100])

def validate_wetdry_spells(mod_d, obs, metric_func, wetspell=True, wet_thres=0.1, months=None):
    
    a_mod_d = mod_d[0]
    
    if months is not None:
        
        mask_season = mod_d['time.month'].to_pandas().isin(months).values
        mask_season = xr.DataArray(mask_season, coords=[mod_d.time])
        mod_d = mod_d.where(mask_season)
        obs = obs.where(mask_season)
    
    if wetspell:
    
        mod_d_bool = mod_d > wet_thres
        obs_bool = obs > wet_thres  
    
    else:
        
        mod_d_bool = mod_d <= wet_thres
        obs_bool = obs <= wet_thres
    
    def calc_spell_metric(da):
    
        spell_metric = np.empty(a_mod_d.shape)
        spell_metric.fill(np.nan)
        rows,cols = np.nonzero((~a_mod_d.isnull()).values)
        
        for r,c, in zip(rows,cols):
            
            spell_lens = runs_of_ones_array(da[:,r,c].values)
            spell_metric[r,c] = metric_func(spell_lens)
        
        spell_metric = xr.DataArray(spell_metric, coords=[mod_d.lat,mod_d.lon], name='spell_metric')
        
        return spell_metric
        
    
    mod_d_spell = calc_spell_metric(mod_d_bool)
    obs_spell = calc_spell_metric(obs_bool)
    
    # Calculate percent and absolute error
    err = mod_d_spell-obs_spell
    pct_err = (err/obs_spell)*100
    pct_err.values[err.values==0] = 0
    
    obs_spell.name = 'obs'
    mod_d_spell.name = 'mod'
    err.name = 'err'
    pct_err.name = 'pct_err'
    
    return xr.merge([err, pct_err, obs_spell, mod_d_spell])

def validate_hss(mod_d, obs, wet_thres=0, months=None):
        
    if months is not None:
        
        mask_season = mod_d['time.month'].to_pandas().isin(months).values
        mod_d = mod_d[mask_season,:,:]
        obs = obs[mask_season,:,:]
    
    obs = obs > wet_thres
    mod_d = mod_d > wet_thres
    
    a = (obs & mod_d).sum(dim='time')
    b = ((~obs) & mod_d).sum(dim='time')
    c = (obs & (~mod_d)).sum(dim='time')
    d = ((~obs) & (~mod_d)).sum(dim='time')
       
    num = 2 * ((a * d) - (b * c))
    denum = ((a + c) * (c + d)) + ((a + b) * (b + d))
    
    hss = num.astype(np.float)/denum 
    hss.name = 'hss'
    return hss

def to_anomalies(da_prcp, base_startend_yrs=None):
    
    da_prcp_mthly = da_prcp.resample('MS', dim='time', how='mean', skipna=False)
    
    if base_startend_yrs is None:
        da_prcp_clim = da_prcp_mthly.groupby('time.month').mean(dim='time') 
    else:
        da_prcp_clim = da_prcp_mthly.loc[base_startend_yrs[0]:base_startend_yrs[1]].groupby('time.month').mean(dim='time')    
    
    da_prcp_anoms = da_prcp.groupby('time.month') / da_prcp_clim

    return da_prcp_anoms

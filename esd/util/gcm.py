import numpy as np
import pandas as pd
import xarray as xr


def unit_convert_pr(da):
    
    # Convert from kg m-2 s-1 to mm per day
    assert da.units == 'kg m-2 s-1'
    da = da*86400
    da.attrs['units'] = 'mm day-1'
    return da

def unit_convert_tas(da):
    
    # Convert from kg m-2 s-1 to mm per day
    assert da.units == 'K'
    da = da-273.15
    da.attrs['units'] = 'C'
    return da

def unit_convert_none(da):
    return da

def clean_gcm_da(da, unit_convert_func=unit_convert_none, startend_dates=None):
    
    # Get rid of duplicated days
    da = da.isel(time=np.nonzero((~da.time.to_pandas().duplicated()).values)[0])

    # Get rid of days > 2101 due to pandas Timestamp range limitation
    da = da.sel(time=da.time[da.time <= 21010101])
    
    # Perform unit conversion
    da = unit_convert_func(da)
    
    # Convert times to datetime format
    times = pd.to_datetime(da.time.to_pandas().astype(np.str),
                           format='%Y%m%d.5', errors='coerce')        
    da['time'] = times.values
    # Get rid of invalid dates (occurs with 360-day calendar)
    da = da.isel(time=np.nonzero(da.time.notnull().values)[0])
    times = da.time.to_pandas()
    
    # Full sequence of valid dates
    # Todo: get 12-31
    first_day = times.iloc[0]
    last_day = times.iloc[-1]
    
    if (last_day.day == 30) and (last_day.month==12):
        last_day = last_day + pd.Timedelta(days=1)
    
    times_full = pd.date_range(first_day, last_day)
    
    # Reindex to full sequence and place NA for missing values
    da = da.reindex(time=times_full)
    
    s = da.to_series() #index: time, lat, lon
    s = s.reorder_levels(['lat','lon','time'])
    s = s.sortlevel(0, sort_remaining=True)
    s = s.unstack(['lat','lon'])
    s = s.interpolate(method='time')
    da_cleaned = xr.DataArray.from_series(s.stack(['lat','lon']))
    da_cleaned.attrs = da.attrs
    
    if startend_dates is not None:
        da_cleaned = da_cleaned.loc[startend_dates[0]:startend_dates[-1]]
    
    return da_cleaned
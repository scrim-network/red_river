'''
Script for and prototyping and testing temperature downscaling approaches.
'''

from esd.downscale.tair import setup_data_for_analog, downscale_analog_anoms,\
    TairDownscale
import esd
import numpy as np
import os
import pandas as pd
import xarray as xr
from esd.downscale.common import _convert_times

if __name__ == '__main__':
    
    fpath_tair_obs = esd.cfg.fpath_aphrodite_tair
    fpath_tair_obsc = os.path.join(esd.cfg.path_aphrodite_resample,
                                   'aprhodite_redriver_sat_1961_2007_p25deg_remapbic.nc')
    fpath_tair_mod = fpath_tair_obsc
    
    base_start_year = '1961'
    base_end_year = '1990'
    downscale_start_year = '1991'
    downscale_end_year = '2007'
        
#     base_start_year = '1978'
#     base_end_year = '2007'
#     downscale_start_year = '1961'
#     downscale_end_year = '1977'
         
    ds_data,win_masks = setup_data_for_analog(fpath_tair_obs, fpath_tair_obsc,
                                              fpath_tair_mod,
                                              base_start_year, base_end_year,
                                              downscale_start_year, downscale_end_year)
     
    mod_downscale_anoms = downscale_analog_anoms(ds_data, win_masks)
    
    tair_d = TairDownscale(fpath_tair_obs, fpath_tair_obsc, base_start_year,
                           base_end_year, [(downscale_start_year,downscale_end_year)])
    da_mod = xr.open_dataset(fpath_tair_mod)
    da_mod['time'] = _convert_times(da_mod)
    mod_downscale_vclim = tair_d.downscale(da_mod.SAT)
         
    da_obsc = xr.open_dataset(fpath_tair_obsc).SAT.load()
    da_obsc['time'] = pd.to_datetime(da_obsc.time.to_pandas().astype(np.str),
                                     format='%Y%m%d', errors='coerce')
    mod_downscale_cg = da_obsc.loc[downscale_start_year:downscale_end_year]
         
    mod_downscale_anoms.name = 'mod_d_anoms'
    mod_downscale_vclim.name = 'mod_d_vclim'
    mod_downscale_cg.name = 'mod_d_cg'
     
    obs = ds_data.obs.loc[downscale_start_year:downscale_end_year]
     
    ds_d = xr.merge([obs, mod_downscale_cg, mod_downscale_anoms, mod_downscale_vclim])
    ds_d = ds_d.drop('month')
    ds_d.to_netcdf(os.path.join(esd.cfg.data_root,'downscaling',
                                'downscale_tests_tair_%s_%s.nc'%(downscale_start_year,
                                                                 downscale_end_year)))

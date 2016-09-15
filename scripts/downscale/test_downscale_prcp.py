'''
Script for and prototyping and testing precipitation downscaling approaches.
'''

from esd.downscale import setup_data_for_analog, downscale_analog_anoms,\
    downscale_analog
import esd
import numpy as np
import os
import pandas as pd
import xarray as xr

if __name__ == '__main__':
    
    fpath_prcp_obs = esd.cfg.fpath_aphrodite_prcp
    fpath_prcp_obsc = os.path.join(esd.cfg.path_aphrodite_resample,
                                   'aprhodite_redriver_pcp_1961_2007_p25deg_remapbic.nc')
    fpath_prcp_mod = fpath_prcp_obsc
    base_start_year = '1961'
    base_end_year = '1990'
    downscale_start_year = '1991'
    downscale_end_year = '2007'
     
    ds_data,win_masks = setup_data_for_analog(fpath_prcp_obs, fpath_prcp_obsc,
                                              fpath_prcp_mod,
                                              base_start_year, base_end_year,
                                              downscale_start_year, downscale_end_year)
     
    mod_downscale_anoms = downscale_analog_anoms(ds_data, win_masks)
    mod_downscale = downscale_analog(ds_data, win_masks)
     
     
    da_obsc = xr.open_dataset(fpath_prcp_obsc).PCP.load()
    da_obsc['time'] = pd.to_datetime(da_obsc.time.to_pandas().astype(np.str),
                                     format='%Y%m%d', errors='coerce')
    mod_downscale_cg = da_obsc.loc[downscale_start_year:downscale_end_year]
         
    mod_downscale_anoms.name = 'mod_d_anoms'
    mod_downscale.name = 'mod_d'
    mod_downscale_cg.name = 'mod_d_cg'
     
    obs = ds_data.obs.loc[downscale_start_year:downscale_end_year]
     
    ds_d = xr.merge([obs, mod_downscale_cg, mod_downscale, mod_downscale_anoms])
    ds_d = ds_d.drop('month')
    ds_d.to_netcdf(os.path.join(esd.cfg.data_root,'downscaling','downscale_tests.nc'))

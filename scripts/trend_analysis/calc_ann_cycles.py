'''
Script to calculate historical annual cycles of tair and prcp for 
APHRODITE and CMIP5 realizations.
'''

import esd
import glob
import itertools
import numpy as np
import os
import pandas as pd
import xarray as xr

if __name__ == '__main__':
    
    # Historical annual cycles for aphrodite 1deg and CMIP5 realizations that
    # have NOT been biased corrected and downscaled
#     path_in_cmip5 = esd.cfg.path_cmip5_cleaned
#     fpath_aphrodite_tair = os.path.join(esd.cfg.path_aphrodite_resample,
#                                         'aprhodite_redriver_sat_1961_2007_1deg.nc')
#     fpath_aphrodite_prcp = os.path.join(esd.cfg.path_aphrodite_resample,
#                                         'aprhodite_redriver_pcp_1961_2007_1deg.nc')
#     fpath_out_tair = os.path.join(esd.cfg.path_cmip5_trends, 'hist_ann_cycle_tair.csv')
#     fpath_out_prcp = os.path.join(esd.cfg.path_cmip5_trends, 'hist_ann_cycle_prcp.csv')
    
    # Historical annual cycles for original aphrodite .25deg and CMIP5 realizations that
    # have been biased corrected and downscaled
    path_in_cmip5 = esd.cfg.path_cmip5_downscaled
    fpath_aphrodite_tair = os.path.join(esd.cfg.path_aphrodite_resample,
                                        'aprhodite_redriver_sat_1961_2007_p25deg.nc')
    fpath_aphrodite_prcp = os.path.join(esd.cfg.path_aphrodite_resample,
                                        'aprhodite_redriver_pcp_1961_2007_p25deg.nc')
    fpath_out_tair = os.path.join(esd.cfg.path_cmip5_trends, 'hist_ann_cycle_tair_p25deg.csv')
    fpath_out_prcp = os.path.join(esd.cfg.path_cmip5_trends, 'hist_ann_cycle_prcp_p25deg.csv')
    
    
    ############################################################################
    
    paths_in = sorted(glob.glob(os.path.join(path_in_cmip5, '*')))
        
    fpaths_tas = sorted(list(itertools.chain.
                             from_iterable([glob.glob(os.path.join(apath,'tas.day*'))
                                            for apath in paths_in])))
    fpaths_pr = sorted(list(itertools.chain.
                            from_iterable([glob.glob(os.path.join(apath,'pr.day*'))
                                           for apath in paths_in])))
    
    ds = xr.open_dataset(fpaths_tas[0])
    mask_lat = np.logical_and(ds.lat.values >= esd.cfg.bbox[1],
                              ds.lat.values <= esd.cfg.bbox[-1])
    mask_lon = np.logical_and(ds.lon.values >= esd.cfg.bbox[0],
                              ds.lon.values <= esd.cfg.bbox[2]) 
    ds.close()
    
    def mth_tair_norms(fpath, vname='tas', apply_mask=True,name=None):
    
        if isinstance(fpath, basestring):
            ds = xr.open_dataset(fpath)
        else:
            ds = fpath
            
        if apply_mask:
            da = ds[vname].loc['1976':'2005',mask_lat,mask_lon].load()
        else:
            da = ds[vname].loc['1976':'2005',:,:].load()
            
        da_mthly = da.resample(freq='MS', dim='time', how='mean', skipna=False)
        norms = da_mthly.groupby('time.month').mean(dim='time').mean(dim=['lat','lon'])
        norms = norms.to_pandas()
        
        if name is None:
            name = os.path.basename(fpath)
        norms.name = name
        ds.close()
        
        print norms.name
        
        return norms
    
    
    norms_tair = [mth_tair_norms(fpath) for fpath in fpaths_tas]
    norms_tair = pd.concat(norms_tair, axis=1)
    
    def mth_prcp_norms(fpath, vname='pr', apply_mask=True,name=None):
        
        if isinstance(fpath, basestring):
            ds = xr.open_dataset(fpath)
        else:
            ds = fpath
            
        if apply_mask:
            da = ds[vname].loc['1976':'2005',mask_lat,mask_lon].load()
        else:
            da = ds[vname].loc['1976':'2005',:,:].load()
            
        da_mthly = da.resample(freq='MS', dim='time', how='sum', skipna=False)
        norms = da_mthly.groupby('time.month').mean(dim='time').mean(dim=['lat','lon'])
        norms = norms.to_pandas()
        if name is None:
            name = os.path.basename(fpath)
        norms.name = name
        ds.close()
        
        print norms.name
        
        return norms
    
    norms_prcp = [mth_prcp_norms(fpath) for fpath in fpaths_pr]
    norms_prcp = pd.concat(norms_prcp, axis=1)
        
    ds_tair = xr.open_dataset(fpath_aphrodite_tair)
    ds_prcp = xr.open_dataset(fpath_aphrodite_prcp)
    
    def convert_times(ds):
        
        if ds.time.values.dtype.name != 'datetime64[ns]':
            # Convert times to datetime format
            times = pd.to_datetime(ds.time.to_pandas().astype(np.str),
                                   format='%Y%m%d.0', errors='coerce')        
            ds['time'] = times.values
            
        return ds
    
    ds_tair = convert_times(ds_tair)
    ds_prcp = convert_times(ds_prcp)
    
    aphro_norms_tair = mth_tair_norms(ds_tair, 'SAT', False, 'aphro')
    aphro_norms_prcp = mth_prcp_norms(ds_prcp, 'PCP', False, 'aphro')
    
    norms_prcp = pd.concat([norms_prcp, aphro_norms_prcp],axis=1)
    norms_tair = pd.concat([norms_tair, aphro_norms_tair],axis=1)
    
    norms_prcp.to_csv(fpath_out_prcp)
    norms_tair.to_csv(fpath_out_tair)
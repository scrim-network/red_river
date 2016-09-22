'''
Script to compute "spatiotemporal correlograms" for daily precipitation.
In the spatiotemporal correlogram, the average correlation
is plotted by pair distance (spatial lag). As distance increases, we expect correlation to decrease.  
The purpose of the spatiotemporal correlograms is to the determine if the
downscaling procedure has any impact on the spatial coherence
of the precipitation field.

See figure 5:
Osborn, T. J., and M. Hulme. 1997. “Development of a Relationship between
Station and Grid-Box Rainday Frequencies for Climate Model Evaluation.”
Journal of Climate 10 (8): 1885–1908.
'''

from esd.downscale import validate_spatial_coherence
import esd
import numpy as np
import os
import pandas as pd
import xarray as xr
from esd.downscale.prcp import to_anomalies
from scipy.stats.stats import spearmanr

if __name__ == '__main__':
    
#    downscale_start_year = '1991'
#    downscale_end_year = '2007'
    downscale_start_year = '1961'
    downscale_end_year = '1977'
    rfunc = lambda x,y: spearmanr(x,y)[0]
    
    ds_d = xr.open_dataset(os.path.join(esd.cfg.data_root,'downscaling',
                                        'downscale_tests_%s_%s.nc'%(downscale_start_year,
                                                                    downscale_end_year))).load()
    mask_null = xr.concat([ds_d[vname].isnull().any(dim='time')
                           for vname in ds_d.data_vars.keys()],'d').any(dim='d')
    for vname in ds_d.data_vars.keys():
        ds_d[vname].values[:,mask_null.values] = np.nan
    
    # Convert to daily anomalies. We want to look at spatiotemporal coherence
    # of anomalies and not the absolute precipitaiton values  
    for vname in ds_d.data_vars.keys():
        ds_d[vname] = to_anomalies(ds_d[vname])
        
    mod_names = ['mod_d_cg','mod_d','mod_d_anoms','mod_d_sgrid','mod_d_anoms_sgrid']
    mod_longnames = ['Coarsened\nAphrodite', 'Analog Using\nAbsolute',
                     'Analog Using\nAnomalies','Analog Using\nAbsolute (Interp Scale)',
                     'Analog Using\nAnomalies (Interp Scale)']
    
    months = [None, [12,1,2], [6, 7, 8, 9]]
    mth_names = ['ann_%s_%s'%(downscale_start_year, downscale_end_year),
                 'dryseason_%s_%s'%(downscale_start_year, downscale_end_year),
                 'monsoonseason_%s_%s'%(downscale_start_year, downscale_end_year)]
    
    path_out = os.path.join(esd.cfg.data_root,'downscaling', 'st_correl')
    
    for a_mth,a_mthname in zip(months,mth_names):
        # Spatial Coherence Annual
        results = {mod_longname:validate_spatial_coherence(ds_d[mod_name],rfunc,months=a_mth).r
                   for mod_longname,mod_name in zip(mod_longnames,mod_names)}
        results['Observed'] = validate_spatial_coherence(ds_d.obs, months=a_mth).r
    
        df_stcorrel = pd.DataFrame(results)
        df_stcorrel.to_csv(os.path.join(path_out,'stcorrel_anomalies_%s.csv'%(a_mthname,)))
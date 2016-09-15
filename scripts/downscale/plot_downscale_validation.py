from esd.downscale import validate_avg_total_prcp
from esd.util import plot_validation_grid_maps
import esd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import xarray as xr

if __name__ == '__main__':
    
    ds_d = xr.open_dataset(os.path.join(esd.cfg.data_root,'downscaling','downscale_tests.nc'))
    
    err_obsc = validate_avg_total_prcp(ds_d.mod_d_cg, ds_d.obs, months=None)
    err = validate_avg_total_prcp(ds_d.mod_d, ds_d.obs, months=None)
    err_anoms = validate_avg_total_prcp(ds_d.mod_d_anoms, ds_d.obs, months=None)
    
    ds_metrics = xr.concat([err_obsc[['mod_avg','err','pct_err']],
                            err[['mod_avg','err','pct_err']],
                            err_anoms[['mod_avg','err','pct_err']]],
                           pd.Index(['Coarsened\nAphrodite',
                                     'Analog Using\nAbsolute',
                                     'Analog Using\nAnomalies'],
                                    name='mod_name'))
    ds_metrics = xr.merge([err.obs_avg,ds_metrics])
    ds_metrics = ds_metrics.rename({'obs_avg':'obs','mod_avg':'mod'})
    ds_metrics.obs.attrs['longname'] = 'Annual Average Total Prcp (mm)'
    ds_metrics.mod.attrs['longname'] = 'Annual Average Total Prcp (mm)'
    ds_metrics.pct_err.attrs['longname'] = 'Percent Error (%)'
    ds_metrics.err.attrs['longname'] = 'Error (mm)'

    grid = plot_validation_grid_maps(ds_metrics, ['pct_err','err'], esd.cfg.bbox)
    for i in [0,3,6,9]:
        cticks = grid.cbar_axes[i].get_xticklabels()
        cticks = np.array(cticks)
        cticks[np.arange(1,cticks.size,step=2)] = ''
        grid.cbar_axes[i].set_xticklabels(cticks)
    
    fig = plt.gcf()
    fig.set_size_inches(9.97,12.44) #w h
    plt.tight_layout()
#     
    plt.savefig(os.path.join(esd.cfg.data_root,'downscaling','figures','ann_total_prcp.png'),
                dpi=150,bbox_inches='tight')
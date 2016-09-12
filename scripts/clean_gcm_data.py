'''
Script to clean GCM data: convert units, convert all calendars to standard
365+leap year.
'''

from esd.util import clean_gcm_da, unit_convert_pr, unit_convert_tas, mkdir_p, StatusCheck
import esd
import glob
import itertools
import numpy as np
import os
import pandas as pd
import xarray as xr

if __name__ == '__main__':

    path_in_cmip5 = esd.cfg.path_cmip5_resample
    path_out_cmip5 = esd.cfg.path_cmip5_cleaned
    paths_in = sorted(glob.glob(os.path.join(path_in_cmip5, '*')))
    paths_out = [os.path.join(path_out_cmip5,os.path.basename(apath)) for apath in paths_in]
    for apath in paths_out:
        mkdir_p(apath)
    
    start_end = (esd.cfg.start_date_baseline, pd.Timestamp('2099-12-31'))
    
    fpaths_tas = sorted(list(itertools.chain.
                             from_iterable([glob.glob(os.path.join(apath,'tas.day*'))
                                            for apath in paths_in])))
    
    fpaths_pr = sorted(list(itertools.chain.
                            from_iterable([glob.glob(os.path.join(apath,'pr.day*'))
                                           for apath in paths_in])))
    
    fpaths_all = np.concatenate((fpaths_pr,fpaths_tas))
    
    def parse_fpath_info(fpath):
        
        try:
        
            bname = os.path.basename(fpath)
            
            if not bname.endswith('.nc'):
                bname = bname + ".nc"
             
            vals = bname.split('.')
            
            elem = vals[0]     
            model_name = vals[2]
            rcp_name = vals[3].split('+')[1]
            realization_name = vals[4]
            
            start_yr,end_yr = [int(i[0:4]) for i in vals[-2].split('-')]
        
        except Exception:
            return tuple([fpath,bname]+[None]*5)
        
        return fpath, bname, elem, model_name, rcp_name, realization_name, start_yr, end_yr
    
    fpaths_df = pd.DataFrame([parse_fpath_info(fpath) for fpath in fpaths_all],
                             columns=['fpath','fname','elem','model_name','rcp_name',
                                      'realization_name','start_yr','end_yr'])
        
    # Remove GCM files with no fpath info or incomplete records of the specified
    # period of interest
    mask_keep = ((fpaths_df.start_yr <= start_end[0].year) &
                 (fpaths_df.end_yr >= start_end[-1].year))
    fpaths_df = fpaths_df[mask_keep]
    
    unit_convert_funcs = {'pr':unit_convert_pr,
                          'tas':unit_convert_tas}
    
    schk = StatusCheck(len(fpaths_df), 10)
    
    for i,fpath_row in fpaths_df.iterrows():
        
        print fpath_row.fpath
        
        ds = xr.open_dataset(fpath_row.fpath)
        da = clean_gcm_da(ds[fpath_row.elem], unit_convert_funcs[fpath_row.elem], start_end)
        ds_out = da.to_dataset(name=fpath_row.elem)
        ds_out.attrs.update(ds.attrs)
        
        fname_out = "%s.day.%s.historical+%s.%s.%s-%s.nc"%(fpath_row.elem,
                                                           fpath_row.model_name,
                                                           fpath_row.rcp_name,
                                                           fpath_row.realization_name,
                                                           start_end[0].strftime('%Y%m%d'),
                                                           start_end[1].strftime('%Y%m%d'))
        fpath_out = os.path.join(path_out_cmip5,
                                 'historical+%s_merged'%fpath_row.rcp_name,
                                 fname_out)
        ds_out.to_netcdf(fpath_out)
        
        schk.increment()
        

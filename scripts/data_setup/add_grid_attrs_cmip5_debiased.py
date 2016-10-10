from esd.util import mkdir_p
import esd
import glob
import os
import xarray as xr

if __name__ == '__main__':
    
    path_in = esd.cfg.path_cmip5_debiased
    path_out = os.path.join(esd.cfg.path_cmip5_debiased, 'with_grid_attrs')
    mkdir_p(path_out)
                
    paths_rcp = sorted(glob.glob(os.path.join(path_in, "historical*_merged")))
    
    for path_rcp in paths_rcp:
        
        fpaths_tas = sorted(glob.glob(os.path.join(path_rcp, 'tas.day*')))
        fpaths_pr = sorted(glob.glob(os.path.join(path_rcp, 'pr.day*')))
        fpaths_all = fpaths_tas + fpaths_pr
        
        path_out_rcp = os.path.join(path_out, os.path.basename(path_rcp))
        mkdir_p(path_out_rcp)
        
        for fpath_in in fpaths_all:
            
            print fpath_in
            
            ds = xr.open_dataset(fpath_in).load()
            ds.lon.attrs['standard_name'] = 'longitude'
            ds.lon.attrs['long_name'] = 'longitude'
            ds.lon.attrs['units'] = 'degrees_east'
            ds.lon.attrs['axis'] = 'X'
            
            ds.lat.attrs['standard_name'] = 'latitude'
            ds.lat.attrs['long_name'] = 'latitude'
            ds.lat.attrs['units'] = 'degrees_north'
            ds.lat.attrs['axis'] = 'Y'
            
            fpath_out = os.path.join(path_out_rcp, os.path.basename(fpath_in))
            ds.to_netcdf(fpath_out)
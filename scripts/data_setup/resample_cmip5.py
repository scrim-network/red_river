'''
Script to resample CMIP5 model data to standard grid domain and resolution.
Uses external calls to cdo.
'''

from esd.util import buffer_cdo_lonlat_grid_cfg, mkdir_p
import esd
import glob
import os
import subprocess

def create_buffered_cdo_grid_file(fpath_cdo_grid_red_river, fpath_out):
    
    grid_cfg = buffer_cdo_lonlat_grid_cfg(fpath_cdo_grid_red_river, buf=6)
    
    with open(fpath_out, 'w') as f:
        f.writelines(grid_cfg)

if __name__ == '__main__':
    
    resample_method = 'remapbil' #remapbil or remapbic
    path_out = esd.cfg.path_cmip5_resample
    fpath_cdo_grid_red_river = os.path.join(esd.cfg.path_data_bbox, 'cdo_grid_1deg_red_river')
    fpath_cdo_target = os.path.join(esd.cfg.path_data_bbox, 'cdo_grid_1deg_red_river_6degbuf')
    
    # If not existing, create CDO grid configuration with a 6 degree buffer
    # around the default Red River 1 deg grid.
    if not os.path.isfile(fpath_cdo_target):
        create_buffered_cdo_grid_file(fpath_cdo_grid_red_river, fpath_cdo_target)
    
    paths_rcp = sorted(glob.glob(os.path.join(esd.cfg.path_cmip5_archive, "historical*_merged")))
    
    for path_rcp in paths_rcp:
        
        fpaths_tas = sorted(glob.glob(os.path.join(path_rcp, 'tas.day*')))
        fpaths_pr = sorted(glob.glob(os.path.join(path_rcp, 'pr.day*')))
        fpaths_all = fpaths_tas + fpaths_pr
        
        path_out_rcp = os.path.join(path_out, resample_method, os.path.basename(path_rcp))
        mkdir_p(path_out_rcp)
        
        for fpath_in in fpaths_all:
            
            fpath_out = os.path.join(path_out_rcp, os.path.basename(fpath_in))
            cmd = "cdo %s,%s %s %s"%(resample_method,fpath_cdo_target, fpath_in, fpath_out)
            print cmd
            subprocess.call(cmd, shell=True)
            
        
        
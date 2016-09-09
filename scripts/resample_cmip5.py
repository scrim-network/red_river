'''
Script to resample CMIP5 model data to standard grid domain and resolution.
Uses external calls to cdo.
'''

import esd
import glob
import os
import subprocess

if __name__ == '__main__':
    
    path_out = esd.cfg.path_cmip5_resample
    fpath_cdo_grid_def = os.path.join(esd.cfg.path_data_bbox, 'cdo_grid_1deg_red_river')
    paths_rcp = sorted(glob.glob(os.path.join(esd.cfg.path_cmip5_archive, "historical*_merged")))
    
    for path_rcp in paths_rcp:
        
        fpaths_tas = sorted(glob.glob(os.path.join(path_rcp, 'tas.day*')))
        fpaths_pr = sorted(glob.glob(os.path.join(path_rcp, 'pr.day*')))
        fpaths_all = fpaths_tas + fpaths_pr
        
        path_out_rcp = os.path.join(path_out, os.path.basename(path_rcp))
        esd.util.mkdir_p(path_out_rcp)
        
        for fpath_in in fpaths_all:
            
            fpath_out = os.path.join(path_out_rcp, os.path.basename(fpath_in))
            cmd = "cdo remapbil,%s %s %s"%(fpath_cdo_grid_def, fpath_in, fpath_out)
            print cmd
            subprocess.call(cmd, shell=True)
            
        
        
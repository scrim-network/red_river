'''
Script to smooth debiased CMIP5 model data from 1deg resolution to the APHRODITE
0.25deg resolution grid. These smoothed versions of the debiased data are used
in the analog downscaling procedure.
'''

from esd.util import mkdir_p
import esd
import glob
import os
import subprocess

if __name__ == '__main__':
    
    # Upsample 1deg debiased CMIP5 datasets to Red River 0.25deg Aphrodite grid with
    # bicubic or bilinear interpolation (remapbil or remapbic)
    # These are used in the analog downscaling procedure
    path_in = os.path.join(esd.cfg.path_cmip5_debiased, 'with_grid_attrs')
    resample_method = 'remapbic' #remapbil or remapbic
    path_out = os.path.join(esd.cfg.path_cmip5_debiased, 'resampled')
    mkdir_p(path_out)
    
    fpath_cdo_grid_def = os.path.join(esd.cfg.path_data_bbox, 'cdo_grid_p25deg_red_river')
    
    def upsample_cdo(fpath_in, fpath_out, method):
        cmd = "cdo %s,%s %s %s"%(method, fpath_cdo_grid_def, fpath_in, fpath_out)
        print cmd
        subprocess.call(cmd, shell=True)
        
    paths_rcp = sorted(glob.glob(os.path.join(path_in, "historical*_merged")))
    
    for path_rcp in paths_rcp:
        
        fpaths_tas = sorted(glob.glob(os.path.join(path_rcp, 'tas.day*')))
        fpaths_pr = sorted(glob.glob(os.path.join(path_rcp, 'pr.day*')))
        fpaths_all = fpaths_tas + fpaths_pr
        
        path_out_rcp = os.path.join(path_out, os.path.basename(path_rcp))
        mkdir_p(path_out_rcp)
        
        for fpath_in in fpaths_all:
            
            fpath_out = os.path.join(path_out_rcp, os.path.basename(fpath_in))
            cmd = "cdo %s,%s %s %s"%(resample_method, fpath_cdo_grid_def, fpath_in, fpath_out)
            print cmd
            subprocess.call(cmd, shell=True)
            
        
        
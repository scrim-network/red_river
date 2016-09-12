'''
Script to resample APHRODITE grids with cdo.
'''

import os
import esd
import subprocess

if __name__ == '__main__':
    
    # Downsample from native 0.25deg to 1deg Red River GCM grid
    fpath_cdo_grid_def = os.path.join(esd.cfg.path_data_bbox, 'cdo_grid_1deg_red_river')
    
    def downsample_cdo(fpath_in, fpath_out):
        cmd = "cdo remapcon,%s %s %s"%(fpath_cdo_grid_def, fpath_in, fpath_out)
        print cmd
        subprocess.call(cmd, shell=True)
    
    fpath_out_1deg_prcp = os.path.join(esd.cfg.path_aphrodite_resample, 'aprhodite_redriver_pcp_1961_2007_1deg.nc')
    downsample_cdo(esd.cfg.fpath_aphrodite_prcp, fpath_out_1deg_prcp)
    
    fpath_out_1deg_tair = os.path.join(esd.cfg.path_aphrodite_resample, 'aprhodite_redriver_sat_1961_2007_1deg.nc')
    downsample_cdo(esd.cfg.fpath_aphrodite_tair, fpath_out_1deg_tair)
        
    # Upsample 1deg downsampled grids to Red River 0.25deg Aphrodite grid with bicubic interpolation
    # These are used in the analog downscaling procedure
    fpath_cdo_grid_def = os.path.join(esd.cfg.path_data_bbox, 'cdo_grid_p25deg_red_river')
    
    def upsample_cdo(fpath_in, fpath_out):
        cmd = "cdo remapbic,%s %s %s"%(fpath_cdo_grid_def, fpath_in, fpath_out)
        print cmd
        subprocess.call(cmd, shell=True)
    
    fpath_out_p25deg_prcp = os.path.join(esd.cfg.path_aphrodite_resample,
                                         'aprhodite_redriver_pcp_1961_2007_p25deg_bicubic.nc')
    upsample_cdo(fpath_out_1deg_prcp, fpath_out_p25deg_prcp)
    
    fpath_out_p25deg_tair = os.path.join(esd.cfg.path_aphrodite_resample,
                                         'aprhodite_redriver_sat_1961_2007_p25deg_bicubic.nc')
    upsample_cdo(fpath_out_1deg_tair, fpath_out_p25deg_tair)     
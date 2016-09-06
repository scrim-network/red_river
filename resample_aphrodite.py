import xarray as xr
import numpy as np
import itertools

'''
CDO Commands
cdo remapcon,/storage/home/jwo118/group_store/red_river/cdo_grid_1deg_red_river /storage/home/jwo118/group_store/red_river/aphrodite/APHRODITE.Monsoon_Asia.SAT.0p25deg.v1204r1.1961-2007.nc remapcon.nc
cdo remapcon,/storage/home/jwo118/group_store/red_river/cdo_grid_1deg_red_river /storage/home/jwo118/group_store/red_river/aphrodite/APHRODITE.Monsoon_Asia.PCP.0p25deg.v1101r2.1961-2007.nc APHRODITE.PCP.0p25deg.remapco
'''


if __name__ == '__main__':
    
    ds_src = xr.open_dataset('/storage/home/jwo118/group_store/red_river/aphrodite/APHRODITE.Monsoon_Asia.PCP.0p25deg.v1101r2.1961-2007.nc')
    ds_target = xr.open_dataset('/storage/group/kzk10_collab/red_river/cmip5_resample/historical+rcp85_merged/tas.day.CCSM4.historical+rcp85.r2i1p1.18500101-21001231.nc')
    
    da = ds_src.PCP[:,139:163,159:187].load()
    
    x = [(i,i+4) for i in np.arange(da.lon.size,step=4)]
    y = [(i,i+4) for i in np.arange(da.lat.size,step=4)]
    
    a = np.empty(ds_target.tas[0].size)
    
    combos = [i for i in itertools.product(y,x)]
    
    for i,r in enumerate(combos):
        a[i] = da[4000,r[0][0]:r[0][1],r[1][0]:r[1][1]].mean()
        
    a = a.reshape(ds_target.tas[0].shape)
    
    a = xr.DataArray(a,coords=[ds_target.tas.lat,ds_target.tas.lon])    
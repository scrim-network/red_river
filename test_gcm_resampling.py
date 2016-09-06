import xarray as xr
import numpy as np
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

BOUNDS_RED_RIVER = 100.0,20.0,106.5833,25.5

'''
cat > mygrid << EOF
gridtype = lonlat
xsize    = 360
ysize    = 180
xfirst   = −179.5
xinc     = 1
yfirst   = -89.5
yinc     = 1
EOF
cdo remapbil,mygrid tas.day.CCSM4.historical+rcp45.r6i1p1.19500101-21001231.nc test.nc
'''


if __name__ == '__main__':
    
    ds = xr.open_dataset('/storage/home/jwo118/group_store/red_river/tas.day.CCSM4.historical+rcp45.r6i1p1.19500101-21001231.nc')
    times = pd.to_datetime(ds.time.to_pandas().astype(np.str), format='%Y%m%d.5')
    ds['time'] = times.values
    
    mask_lat = (ds.lat >= BOUNDS_RED_RIVER[1]) & (ds.lat <= BOUNDS_RED_RIVER[3]) 
    mask_lon = (ds.lon >= BOUNDS_RED_RIVER[0]) & (ds.lon <= BOUNDS_RED_RIVER[2])
    
    a = ds.tas.loc['1980-05-14', mask_lat, mask_lon].load()
    
    a = ds.tas.loc['1980-05-14'].load()
    a.values[~mask_lat.values,:] = np.nan
    a.values[:,~mask_lon.values] = np.nan
    
    
    # Setup map
    lon0 = a.lon.values.mean()
    ax = plt.axes(projection=ccrs.PlateCarree(lon0))
    g = a.plot.pcolormesh(transform=ccrs.PlateCarree())
    

    buf = 10    
    ax.set_extent(extents=(BOUNDS_RED_RIVER[0]-buf,BOUNDS_RED_RIVER[2]+buf,BOUNDS_RED_RIVER[1]-buf,BOUNDS_RED_RIVER[3]+buf))
    
    #ax.coastlines()
    
    # Coast and border features
    coasts = cartopy.feature.NaturalEarthFeature('physical',
                                                 'coastline',
                                                  scale='50m',edgecolor='black',
                                                  facecolor='none')
    
    borders = cartopy.feature.NaturalEarthFeature('cultural',
                                                  'admin_0_boundary_lines_land',
                                                  scale='50m',edgecolor='black',
                                                  facecolor='none')
    
    ax.add_feature(coasts)
    ax.add_feature(borders)

import cartopy
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("darkgrid")
sns.set_context('paper')


BOUNDS_RED_RIVER = 100.0,20.0,106.5833,25.5

BOUNDS_RED_RIVER_APHRODITE = 99.875,19.875,106.625,25.625

# Open APHRODITE netcdf file
ds = xr.open_dataset('/storage/home/jwo118/group_store/red_river/aphrodite/APHRODITE.Monsoon_Asia.PCP.0p25deg.v1101r2.1961-2007.nc')
vname = 'PCP'

ds = xr.open_dataset('/storage/home/jwo118/group_store/red_river/aphrodite/APHRODITE.Monsoon_Asia.SAT.0p25deg.v1204r1.1961-2007.nc')
vname = 'SAT'



mask_lat = (ds.lat >= BOUNDS_RED_RIVER[1]) & (ds.lat <= BOUNDS_RED_RIVER[3]) 
mask_lon = (ds.lon >= BOUNDS_RED_RIVER[0]) & (ds.lon <= BOUNDS_RED_RIVER[2])

# Time coordinates not properly decoded (non-standard units?)
# Manually change to datetime
times = pd.to_datetime(ds.time.to_pandas().astype(np.str), format='%Y%m%d.0')
ds['time'] = times.values

# Subset to bounding box and load data into memory
# Will take minute or so to load
da = ds[vname][:,mask_lat,mask_lon].load()

# Prcp
da_mthly = da.resample(freq='MS', dim='time', how='sum', skipna=False)
plt.plot(np.arange(1,13),da_mthly.groupby('time.month').mean().values,'.-',ms=10)
plt.xlabel('Month')
plt.ylabel('Precipitation (mm)')
plt.xlim((1,12))
plt.xticks(np.arange(1,13))

# Tair
da_mthly = da.resample(freq='MS', dim='time', how='mean')
plt.plot(np.arange(1,13),da_mthly.groupby('time.month').mean().values,'.-',ms=10)
plt.xlabel('Month')
plt.ylabel('Temperature (degC)')
plt.xlim((1,12))
plt.xticks(np.arange(1,13))

# Prcp
da_ann_norm = da_mthly.resample(freq='A', dim='time', how='sum', skipna=False).mean(dim='time')

# Temp
da_ann_norm = da_mthly.resample(freq='A', dim='time', how='mean').mean(dim='time')

# Setup map
lon0 = da_ann_norm.lon.values.mean()
ax = plt.axes(projection=ccrs.PlateCarree(lon0))
g = da_ann_norm.plot.pcolormesh(transform=ccrs.PlateCarree())


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


# Calculate the monthly normals for sat and pcp
sat_norms = (sat.resample(freq='MS', dim='time', how='mean').
             groupby('time.month').mean(dim='time'))

pcp_norms = (pcp.resample(freq='MS', dim='time', how='sum', skipna=False).
             groupby('time.month').mean(dim='time'))

# Setup map

# Coast and border features
coasts = cartopy.feature.NaturalEarthFeature('physical',
                                             'coastline',
                                              scale='50m',edgecolor='black',
                                              facecolor='none')

borders = cartopy.feature.NaturalEarthFeature('cultural',
                                              'admin_0_boundary_lines_land',
                                              scale='50m',edgecolor='black',
                                              facecolor='none')

lon0 = sat.lon.values.mean()

# PCP map
g = pcp_norms.plot.pcolormesh(col='month', col_wrap=4, transform=ccrs.PlateCarree(),
                              subplot_kws=dict(projection=ccrs.PlateCarree(lon0)))
for ax in g.axes.flat:
    ax.add_feature(coasts)
    ax.add_feature(borders)
    ax.set_extent(bbox)
    ax.axis('off')

# SAT map
g = sat_norms.plot.pcolormesh(col='month', col_wrap=4, transform=ccrs.PlateCarree(),
                              subplot_kws=dict(projection=ccrs.PlateCarree(lon0)))
for ax in g.axes.flat:
    ax.add_feature(coasts)
    ax.add_feature(borders)
    ax.set_extent(bbox)
    ax.axis('off')

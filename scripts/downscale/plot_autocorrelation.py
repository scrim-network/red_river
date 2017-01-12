from scipy.interpolate import splrep,splev
import esd
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr
from pandas.tools.plotting import autocorrelation_plot
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from esd.util.plot import get_levels

if __name__ == '__main__':

#     downscale_start_year = '1961'
#     downscale_end_year = '1977'

    downscale_start_year = '1991'
    downscale_end_year = '2007'
    
    fpath_ds_d = os.path.join(esd.cfg.data_root,'downscaling',
                              'downscale_tests_%s_%s.nc'%(downscale_start_year,
                                                          downscale_end_year))
    
    ds_d = xr.open_dataset(fpath_ds_d).load()
    mask_null = xr.concat([ds_d[vname].isnull().any(dim='time')
                           for vname in ds_d.data_vars.keys()],'d').any(dim='d')
    for vname in ds_d.data_vars.keys():
        ds_d[vname].values[:,mask_null.values] = np.nan
        
    r = 10
    c = 10
        
    obs = ds_d.obs[:,r,c]
    mod = ds_d.mod_d_anoms[:,r,c]
    
    def detrend(a):
        
        a_mthly = a.resample('MS', dim='time', how='mean', skipna=False)
        a_clim = a_mthly.groupby('time.month').mean(dim='time') 
        a_anoms = a.groupby('time.month') / a_clim
        a_anoms = a_anoms.drop('month')
        
        return a_anoms
    
    obs_dt = detrend(obs)
    mod_dt = detrend(mod)
    
    obs_wet = (obs > 0).astype(np.int)
    mod_wet = (mod > 0).astype(np.int)
    
    plt.figure()
    autocorrelation_plot(obs,label='obs')
    autocorrelation_plot(mod,label='mod')
    plt.xlim(0,30)
    plt.ylim(0,.7)
    plt.title('Prcp Amount')
    
    plt.figure()
    autocorrelation_plot(obs_wet,label='obs')
    autocorrelation_plot(mod_wet,label='mod')
    plt.xlim(0,30)
    plt.ylim(0,.7)
    plt.title('Prcp Occurrence')
    
    plt.figure()
    obs_wet.resample('MS', dim='time', how='sum').groupby('time.month').mean(dim='time').plot(label='obs')
    mod_wet.resample('MS', dim='time', how='sum').groupby('time.month').mean(dim='time').plot(label='mod')
    plt.xlim(1,12)
    plt.ylabel('# of days of prcp')
    plt.legend()
    
    plt.figure()
    obs.resample('MS', dim='time', how='sum').groupby('time.month').mean(dim='time').plot(label='obs')
    mod.resample('MS', dim='time', how='sum').groupby('time.month').mean(dim='time').plot(label='mod')
    plt.xlim(1,12)
    plt.ylabel('prcp amt (mm)')
    plt.legend()
    
    
    x = np.arange(obs.size)
    y = obs.values
    
    tck = splrep(x, y, s=1000000)
    
    y_s = splev(x,tck,der=0)
    
    plt.plot(y)
    plt.plot(y_s)
    plt.show()
    
    
    # APHRODITE PRCP OCCUR Issues
    ds = xr.open_dataset('/storage/home/jwo118/scratch/red_river/aphrodite_resample/aprhodite_redriver_pcp_1961_2007_p25deg.nc')
    obs = ds.PCP[:,r,c]
    obs_wet = (obs > 0).astype(np.int)
    ndays_wet = obs_wet.resample('MS', dim='time', how='sum')
    ndays_wet[ndays_wet['time.month']==12].plot()
    plt.ylabel('# of wet days (December)')
    fig = plt.gcf()
    #fig.set_size_inches(9.97,(12.44/4)*(len(mod_names)+1)) #w h
    plt.tight_layout()
    plt.savefig(os.path.join(esd.cfg.data_root, 'downscaling', 'figures','wet_days_figs','nwet_days_dec_tseries.png'), dpi=150, bbox_inches='tight')
    
    
    # 1961 - 1980
    # 1981 - 2007
    plt.figure()
    obs_wet1 = (ds.PCP.loc['1961':'1980'] > 0).resample('MS', dim='time', how='sum',skipna=False).astype(np.float)
    na_mask = ds.PCP.loc['1961':'1980'].isnull().sum(dim='time') != 0
    obs_wet1.values[:,na_mask.values] = np.nan
    obs_wet1.groupby('time.month').mean(dim='time')[-1].plot()
    
    plt.figure()
    obs_wet2 = (ds.PCP.loc['1981':'2007'] > 0).resample('MS', dim='time', how='sum',skipna=False).astype(np.float)
    na_mask = ds.PCP.loc['1981':'2007'].isnull().sum(dim='time') != 0
    obs_wet2.values[:,na_mask.values] = np.nan
    obs_wet2.groupby('time.month').mean(dim='time')[-1].plot()
    
    
    ################################
    
    lon = ds.lon
    lat = ds.lat
    bbox = esd.cfg.bbox
 
    buf = .5
    m = Basemap(resolution='i',projection='cyl', llcrnrlat=bbox[1]-buf,
                area_thresh=100000, urcrnrlat=bbox[-1]+buf,
                llcrnrlon=bbox[0]-buf, urcrnrlon=bbox[2]+buf)
     
    x_map, y_map = m(*np.meshgrid(lon,lat))
     
    axes_pad = .2
    cfig = plt.gcf()
    grid = ImageGrid(cfig,111,nrows_ncols=(1,2),
                     axes_pad=axes_pad, cbar_mode="single", cbar_pad=.05, cbar_location="bottom",
                     cbar_size='3%')
    
    nwet1 = obs_wet1.groupby('time.month').mean(dim='time')[-1]
    nwet2 = obs_wet2.groupby('time.month').mean(dim='time')[-1]
    a = np.concatenate((nwet1.values.ravel(),nwet2.values.ravel()))
    a = a[np.isfinite(a)]
    levels = get_levels(a, diverging=False)
    cmap = 'viridis'
    
    m.ax = grid[0]
    m.ax.set_title('December: 1961-1980')
    m.drawcountries()
    m.drawcoastlines()
    cntr = m.contourf(x_map,y_map,nwet1.values,levels=levels, cmap=cmap, extend='both')
    m.ax = grid[1]
    m.ax.set_title('December: 1981-2007')
    m.drawcountries()
    m.drawcoastlines()
    cntr = m.contourf(x_map,y_map,nwet2.values,levels=levels, cmap=cmap, extend='both')
    
    cbar = plt.colorbar(cntr, cax = grid.cbar_axes[0],orientation='horizontal',extend='both')
    #cbar.ax.tick_params(labelsize=8) 
    cbar.ax.set_xlabel('average # of wet days')
    
    fig = plt.gcf()
    #fig.set_size_inches(9.97,(12.44/4)*(len(mod_names)+1)) #w h
    plt.tight_layout()
    plt.savefig(os.path.join(esd.cfg.data_root, 'downscaling', 'figures','wet_days_figs','map_nwet_days_dec.png'), dpi=150, bbox_inches='tight')
    
    ################################
    

    (obs_wet2.groupby('time.month').mean(dim='time')[-1]-obs_wet1.groupby('time.month').mean(dim='time')[-1]).plot()
    
    
    
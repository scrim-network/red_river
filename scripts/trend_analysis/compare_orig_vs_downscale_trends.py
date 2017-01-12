import esd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap
from esd.util.plot import get_levels
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
import seaborn as sns

if __name__ == '__main__':
    
    fpath_trend_csv_orig = os.path.join(esd.cfg.path_cmip5_trends,'orig' ,'red_river_trends.csv')
    fpath_trend_csv_bc = os.path.join(esd.cfg.path_cmip5_trends,'bias_corrected' ,'red_river_trends.csv')
    fpath_trend_csv_ds = os.path.join(esd.cfg.path_cmip5_trends,'downscaled' ,'red_river_trends.csv')
    
    df_orig = pd.read_csv(fpath_trend_csv_orig)
    df_bc = pd.read_csv(fpath_trend_csv_bc)
    df_ds = pd.read_csv(fpath_trend_csv_ds)
    
    season = 'ANNUAL'
    elem = 'pr'
    time_period = '2006_2099'
    
    def subset_trends(df):
        
        mask = (df.season == season) & (df.elem == elem) & (df.time_period == time_period)
        df = df[mask]
        df = df.set_index('model_run')
        return df[['trend']]
    
    df_orig1 = subset_trends(df_orig)
    df_bc = subset_trends(df_bc)
    df_ds = subset_trends(df_ds)
    
    df_all = df_orig1.join(df_bc,rsuffix='_bc',lsuffix='_orig')
    df_all = df_all.join(df_ds)
    df_all = df_all.rename(columns={'trend':'trend_ds'})
    
    fig,axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    
    plt.sca(axes[0])
    plt.plot(df_all.trend_orig,df_all.trend_bc,'.')
    plt.title("Bias Corrected vs. Original")
    plt.xlabel("Original Trend (% per decade)")
    plt.ylabel("Bias Corrected Trend (% per decade)")
    
    plt.sca(axes[1])
    plt.plot(df_all.trend_orig,df_all.trend_ds,'.',label='model run')
    plt.title("Downscaled vs Original")
    plt.xlabel("Original Trend (% per decade)")
    plt.ylabel("Downscaled Trend (% per decade)")
        
    plt.sca(axes[2])
    plt.plot(df_all.trend_bc,df_all.trend_ds,'.',label='model run')
    plt.title("Downscaled vs Bias Corrected")
    plt.xlabel("Downscaled Trend (% per decade)")
    plt.ylabel("Bias Corrected Trend (% per decade)")        

    ################
    
    plt.sca(axes[0])
    x = np.linspace(plt.xlim()[0], plt.xlim()[-1], num=500)
    plt.plot(x,x,color='#e41a1c')
    
    plt.sca(axes[1])
    x = np.linspace(plt.xlim()[0], plt.xlim()[-1], num=500)
    plt.plot(x,x,color='#e41a1c')
    
    plt.sca(axes[2])
    x = np.linspace(plt.xlim()[0], plt.xlim()[-1], num=500)
    plt.plot(x,x,color='#e41a1c', label='1:1 line')
    plt.legend(loc=4)

    plt.suptitle("Monsoon Season Precipitation Trend: 2006-2099")
    
    ###########################################################################
    
    fpath_trend_csv_orig = os.path.join(esd.cfg.path_cmip5_trends,'orig' ,'red_river_trends.csv')
    fpath_trend_csv_bc_fixed = os.path.join(esd.cfg.path_cmip5_trends,'bias_corrected' ,'red_river_trends.csv')
    fpath_trend_csv_bc =  os.path.join(esd.cfg.path_cmip5_trends, 'bias_corrected','run1', 'red_river_trends.csv')

    df_orig = pd.read_csv(fpath_trend_csv_orig)
    df_bc = pd.read_csv(fpath_trend_csv_bc)
    df_bc_fixed = pd.read_csv(fpath_trend_csv_bc_fixed)
    
    season = 'JJAS'
    elem = 'pr'
    time_period = '2006_2099'
    
    def subset_trends(df):
        
        mask = (df.season == season) & (df.elem == elem) & (df.time_period == time_period)
        df = df[mask]
        df = df.set_index('model_run')
        return df[['trend']]
    
    df_orig1 = subset_trends(df_orig)
    df_bc = subset_trends(df_bc)
    df_bcf = subset_trends(df_bc_fixed)
    
    df_all = df_orig1.join(df_bc,rsuffix='_bc',lsuffix='_orig')
    df_all = df_all.join(df_bcf)
    df_all = df_all.rename(columns={'trend':'trend_bcf'})
    
    fig,axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    
    plt.sca(axes[0])
    plt.plot(df_all.trend_orig,df_all.trend_bc,'.')
    plt.title("Bias Corrected v1")
    plt.xlabel("Original Trend (% per decade)")
    plt.ylabel("Bias Corrected Trend (% per decade)")
    
    plt.sca(axes[1])
    plt.plot(df_all.trend_orig,df_all.trend_bcf,'.',label='model run')
    plt.title("Bias Corrected v2")
    plt.xlabel("Original Trend (% per decade)")
    #plt.ylabel("Bias Corrected Trend (% per decade)")
        
    ################
    
    plt.sca(axes[0])
    x = np.linspace(plt.xlim()[0], plt.xlim()[-1], num=500)
    plt.plot(x,x,color='#e41a1c')
    
    plt.sca(axes[1])
    x = np.linspace(plt.xlim()[0], plt.xlim()[-1], num=500)
    plt.plot(x,x,color='#e41a1c', label='1:1 line')
    plt.legend(loc=4)

    plt.suptitle("Monsoon Season Precipitation Trend: 2006-2099")

    #############
    
    fpath_trend_csv_orig = os.path.join(esd.cfg.path_cmip5_trends,'orig' ,'red_river_trends.csv')
    fpath_trend_csv_bc = os.path.join(esd.cfg.path_cmip5_trends,'bias_corrected' ,'red_river_trends.csv')
    fpath_trend_csv_ds = os.path.join(esd.cfg.path_cmip5_trends, 'downscaled', 'red_river_trends.csv')
        
    df_orig = pd.read_csv(fpath_trend_csv_orig)
    df_bc = pd.read_csv(fpath_trend_csv_bc)
    df_ds = pd.read_csv(fpath_trend_csv_ds)
    
    season = 'DJF'
    elem = 'pr'
    time_period = '2006_2099'
    
    def subset_trends(df):
        
        mask = (df.season == season) & (df.elem == elem) & (df.time_period == time_period)
        df = df[mask]
        df = df.set_index('model_run')
        return df[['trend']]
    
    df_orig1 = subset_trends(df_orig)
    df_bc1 = subset_trends(df_bc)
    df_ds1 = subset_trends(df_ds)
    
    df_all = df_orig1.join(df_bc1,rsuffix='_bc',lsuffix='_orig')
    df_all = df_all.join(df_ds1)
    df_all = df_all.rename(columns={'trend':'trend_ds'})
    
    # Pairs
    # 1.) Orig vs. BC
    # 2.) Orig vs. DS
    # 3.) BC vs. DS    
    
    fig,axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    
    plt.sca(axes[0])
    plt.plot(df_all.trend_orig,df_all.trend_bc,'.')
    plt.title("ORIG vs. BC")
    
    plt.sca(axes[1])
    plt.plot(df_all.trend_orig,df_all.trend_ds,'.')
    plt.title("ORIG vs. DS")
    
    plt.sca(axes[2])
    plt.plot(df_all.trend_bc, df_all.trend_ds,'.')
    plt.title("BC vs. DS")
    
    ################
    
    plt.sca(axes[0])
    x = np.linspace(plt.xlim()[0], plt.xlim()[-1], num=500)
    plt.plot(x,x,color='red')
    
    plt.sca(axes[1])
    x = np.linspace(plt.xlim()[0], plt.xlim()[-1], num=500)
    plt.plot(x,x,color='red')
    
    plt.sca(axes[2])
    x = np.linspace(plt.xlim()[0], plt.xlim()[-1], num=500)
    plt.plot(x,x,color='red')
    
    # Investigate IPSL-CM5A-MR_rcp26_r1i1p1
    
    ds_orig = xr.open_dataset('/storage/home/jwo118/scratch/red_river/cmip5_cleaned/historical+rcp26_merged/pr.day.IPSL-CM5A-MR.historical+rcp26.r1i1p1.19760101-20991231.nc')
    ds_bc = xr.open_dataset('/storage/home/jwo118/scratch/red_river/cmip5_debiased/historical+rcp26_merged/pr.day.IPSL-CM5A-MR.historical+rcp26.r1i1p1.19760101-20991231.nc')
    ds_ds = xr.open_dataset('/storage/home/jwo118/scratch/red_river/cmip5_downscaled/historical+rcp26_merged/pr.day.IPSL-CM5A-MR.historical+rcp26.r1i1p1.19760101-20991231.nc')
    mask_lat = np.logical_and(ds_orig.lat.values >= esd.cfg.bbox[1],
                              ds_orig.lat.values <= esd.cfg.bbox[-1])
    mask_lon = np.logical_and(ds_orig.lon.values >= esd.cfg.bbox[0],
                              ds_orig.lon.values <= esd.cfg.bbox[2]) 
    da_orig = ds_orig.pr[:, mask_lat, mask_lon].load()
    da_bc = ds_bc.pr[:, mask_lat, mask_lon].load()

    da_orig.resample('AS', dim='time', how='sum').mean(dim=['lon','lat']).plot()
    da_bc.resample('AS', dim='time', how='sum').mean(dim=['lon','lat']).plot()
    
    da_orig_mthly = da_orig.resample('MS', dim='time', how='sum')
    da_bc_mthly = da_bc.resample('MS', dim='time', how='sum')
    da_ds_mthly = ds_ds.pr.resample('MS', dim='time', how='sum')
    
    da_ds_mthly.loc['2019-12-01'].plot()
    
    lon = da_ds_mthly.lon
    lat = da_ds_mthly.lat
    bbox = esd.cfg.bbox
 
    buf = .5
    m = Basemap(resolution='i',projection='cyl', llcrnrlat=bbox[1]-buf,
                area_thresh=100000, urcrnrlat=bbox[-1]+buf,
                llcrnrlon=bbox[0]-buf, urcrnrlon=bbox[2]+buf)
     
    x_map, y_map = m(*np.meshgrid(lon,lat))
     
    axes_pad = .2
    cfig = plt.gcf()
    grid = ImageGrid(cfig,111,nrows_ncols=(1,1),
                     axes_pad=axes_pad, cbar_mode="single", cbar_pad=.05, cbar_location="bottom",
                     cbar_size='3%')
    levels = get_levels(da_ds_mthly.loc['2019-12-01'].values.ravel(), diverging=False)
    cmap = 'viridis'
    m.ax = grid[0]
    cntr = m.contourf(x_map,y_map,da_ds_mthly.loc['2019-12-01'].values, cmap=cmap)
    cbar = plt.colorbar(cntr, cax = grid.cbar_axes[0],orientation='horizontal')
    m.ax.set_title('Prcp Total: December 2019\nIPSL-CM5A-MR_rcp26_r1i1p1')
    m.drawcountries()
    m.drawcoastlines()
    cbar.ax.set_xlabel('mm')
    
    fig = plt.gcf()
    plt.tight_layout()
    plt.savefig(os.path.join(esd.cfg.data_root, 'downscaling', 'figures','wet_days_figs','debias_delta_issue.png'), dpi=150, bbox_inches='tight')
    
    
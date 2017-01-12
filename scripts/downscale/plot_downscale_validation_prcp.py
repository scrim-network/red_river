from esd.downscale.common import validate_quantile, validate_correlation,\
    validate_temporal_sd, validate_spatial_sd_monthly
from esd.downscale.prcp import validate_avg_total_prcp, validate_avg_wet_day_frac,\
    validate_wetdry_spells, validate_hss, to_anomalies
from esd.util import plot_validation_grid_maps, get_levels, mkdir_p
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from mpl_toolkits.basemap import Basemap
from scipy.stats.stats import spearmanr, pearsonr
import esd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import xarray as xr

def plot_avg_total_prcp(ds_d, mod_names, mod_longnames, months=None, fpath_out=None):
    
    err_ls = [validate_avg_total_prcp(ds_d[mod_name], ds_d.obs, months=months)
              for mod_name in mod_names]
        
    ds_metrics = xr.concat([a_err[['mod_avg','err','pct_err']] for a_err in err_ls],
                           pd.Index(mod_longnames, name='mod_name'))
    ds_metrics = xr.merge([err_ls[0].obs_avg,ds_metrics])
    ds_metrics = ds_metrics.rename({'obs_avg':'obs','mod_avg':'mod'})
    
    if months == None:
        title = 'Annual Average Total Prcp (mm)'
    else:
        title = "Months "+str(months)+": Average Total Prcp (mm)"
    
    ds_metrics.obs.attrs['longname'] = title
    ds_metrics.mod.attrs['longname'] = title
    ds_metrics.pct_err.attrs['longname'] = 'Percent Error (%)'
    ds_metrics.err.attrs['longname'] = 'Error (mm)'

    grid = plot_validation_grid_maps(ds_metrics, ['pct_err','err'], esd.cfg.bbox)
    for i in np.arange((len(mod_names)*3) + 3 ):# [0,3,6,9]:
        cticks = grid.cbar_axes[i].get_xticklabels()
        cticks = np.array(cticks)
        cticks[np.arange(1,cticks.size,step=2)] = ''
        grid.cbar_axes[i].set_xticklabels(cticks)
        grid.cbar_axes[i].tick_params(pad=-5)
    
    fig = plt.gcf()
    #fig.set_size_inches(9.97,(12.44/4)*(len(mod_names)+1)) #w h
    fig.set_size_inches(7.86,7.5) #w h
    plt.tight_layout()
     
    if fpath_out is not None:
        plt.savefig(fpath_out, dpi=300, bbox_inches='tight')

def plot_avg_wetday_frac(ds_d, mod_names, mod_longnames, wet_thres=0.1, months=None, fpath_out=None):
    
    err_ls = [validate_avg_wet_day_frac(ds_d[mod_name], ds_d.obs, wet_thres, months)
              for mod_name in mod_names]
    
    ds_metrics = xr.concat([a_err[['mod_avg','err']] for a_err in err_ls],
                           pd.Index(mod_longnames, name='mod_name'))
    ds_metrics = xr.merge([err_ls[0].obs_avg,ds_metrics])
    
    ds_metrics = ds_metrics.rename({'obs_avg':'obs','mod_avg':'mod'})
    ds_metrics.obs.attrs['longname'] = 'Wet Day Percentage (%)' + "\n%.2f mm threshold"%wet_thres
    ds_metrics.mod.attrs['longname'] = 'Wet Day Percentage (%)' + "\n%.2f mm threshold"%wet_thres
    ds_metrics.err.attrs['longname'] = 'Error (pp)'

    grid = plot_validation_grid_maps(ds_metrics, ['err'], esd.cfg.bbox)
    for i in np.arange((len(mod_names)*2) + 2 ):
        cticks = grid.cbar_axes[i].get_xticklabels()
        cticks = np.array(cticks)
        cticks[np.arange(1,cticks.size,step=2)] = ''
        grid.cbar_axes[i].set_xticklabels(cticks)
        grid.cbar_axes[i].tick_params(pad=-5)
    
    fig = plt.gcf()
    fig.set_size_inches(7.86,7.5)
    plt.tight_layout()
     
    if fpath_out is not None:
        plt.savefig(fpath_out, dpi=300, bbox_inches='tight')

def plot_quantile(ds_d, mod_names, mod_longnames, q, months=None, fpath_out=None):
        
    err_ls = [validate_quantile(ds_d[mod_name], ds_d.obs, q, months)
              for mod_name in mod_names]
    
    ds_metrics = xr.concat([a_err[['mod','pct_err','err']] for a_err in err_ls],
                           pd.Index(mod_longnames, name='mod_name'))
    ds_metrics = xr.merge([err_ls[0].obs,ds_metrics])
    ds_metrics.obs.attrs['longname'] = '%.1f Percentile (mm)'%q
    ds_metrics.mod.attrs['longname'] = '%.1f Percentile (mm)'%q
    ds_metrics.pct_err.attrs['longname'] = 'Percent Error (%)'
    ds_metrics.err.attrs['longname'] = 'Error (mm)'

    grid = plot_validation_grid_maps(ds_metrics, ['pct_err', 'err'], esd.cfg.bbox)
    for i in np.arange((len(mod_names)*3) + 3 ):
        cticks = grid.cbar_axes[i].get_xticklabels()
        cticks = np.array(cticks)
        cticks[np.arange(1,cticks.size,step=2)] = ''
        grid.cbar_axes[i].set_xticklabels(cticks)
        grid.cbar_axes[i].tick_params(pad=-5)
    
    fig = plt.gcf()
    fig.set_size_inches(7.86,7.5) #w h
    plt.tight_layout()
     
    if fpath_out is not None:
        plt.savefig(fpath_out, dpi=300, bbox_inches='tight')
    
def plot_correlation(ds_d, mod_names, mod_longnames, rfunc, title, months=None, fpath_out=None):
        
    correl_ls = [validate_correlation(ds_d[mod_name], ds_d.obs, rfunc, months)
                 for mod_name in mod_names]
            
    lon = correl_ls[0].lon
    lat = correl_ls[0].lat
 
    buf = .5
    bbox = esd.cfg.bbox
    m = Basemap(resolution='i',projection='cyl', llcrnrlat=bbox[1]-buf,
                area_thresh=100000, urcrnrlat=bbox[-1]+buf,
                llcrnrlon=bbox[0]-buf, urcrnrlon=bbox[2]+buf)
     
    x_map, y_map = m(*np.meshgrid(lon,lat))
     
    axes_pad = .4
    cfig = plt.gcf()
    
    nrow = int(np.ceil(len(mod_names)/3.0))
    grid = ImageGrid(cfig,111,nrows_ncols=(nrow, 3),
                     axes_pad=axes_pad, cbar_mode="each", cbar_pad=.1, cbar_location="bottom",
                     cbar_size='4%')
    
    levels = get_levels(np.concatenate([a_correl.values.ravel() for a_correl in correl_ls]), diverging=False)
                  
    #TODO: make dynamic     
    lon_txt = 99.8
    lat_txt = 25.75
    x_txt,y_txt = m(lon_txt, lat_txt)
    
    def contour_da(da, i, title):
    
        m.ax = grid[i]
        cntr = m.contourf(x_map,y_map,da.values,levels=levels, cmap='viridis', extend='both')
        cbar = plt.colorbar(cntr, cax = grid.cbar_axes[i],orientation='horizontal',extend='both')
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_xlabel('r',fontsize=8) 
        m.drawcountries()
        m.drawcoastlines()
        m.ax.text(x_txt,y_txt,'Mean: %.2f'%(da.mean(),),fontsize=8)
        m.ax.text(x_txt+2,y_txt,title,fontsize=8)
        #grid[i].set_title(title, fontsize=8)
    
    i = 0
    
    for a_correl,a_modname in zip(correl_ls, mod_longnames):
        contour_da(a_correl, i,a_modname)
        i+=1
        
    for j in np.arange(i,len(grid)):
        m.ax = grid[j]
        m.ax.axis('off')
        grid.cbar_axes[j].axis('off')
        
    
    for i in np.arange(len(mod_names)):
        cticks = grid.cbar_axes[i].get_xticklabels()
        cticks = np.array(cticks)
        cticks[np.arange(1,cticks.size,step=2)] = ''
        grid.cbar_axes[i].set_xticklabels(cticks)
    
    
    plt.suptitle(title)
    
    fig = plt.gcf()
    fig.set_size_inches(16.350,(11.52/2)*(nrow)) #w h
    plt.tight_layout()
     
    if fpath_out is not None:
        plt.savefig(fpath_out, dpi=150, bbox_inches='tight')
    
    return grid
    
def plot_st_correlogram(df_stcorrel, mod_longnames, fpath_out=None):
    
    with plt.style.context(('seaborn-whitegrid')):
        
        df_stcorrel[['Observed']+mod_longnames].plot()
        plt.xlabel('Distance Lag (km)')
        plt.ylabel("Spearman Coefficient")
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        if fpath_out is not None:
            plt.savefig(fpath_out, dpi=150, bbox_inches='tight')

def plot_wetspell(ds_d, mod_names, mod_longnames, metric_func, title, wet_thres=.1, months=None, fpath_out=None):
        
    err_ls = [validate_wetdry_spells(ds_d[mod_name], ds_d.obs,
                                     metric_func=metric_func, wetspell=True,
                                     wet_thres=wet_thres, months=months)
              for mod_name in mod_names]
    
    ds_metrics = xr.concat([a_err[['mod','pct_err','err']] for a_err in err_ls],
                           pd.Index(mod_longnames, name='mod_name'))
    ds_metrics = xr.merge([err_ls[0].obs, ds_metrics])
    
    ds_metrics.obs.attrs['longname'] = title
    ds_metrics.mod.attrs['longname'] = title
    ds_metrics.pct_err.attrs['longname'] = 'Percent Error (%)'
    ds_metrics.err.attrs['longname'] = 'Error (days)'

    grid = plot_validation_grid_maps(ds_metrics, ['pct_err', 'err'], esd.cfg.bbox)
    
    fig = plt.gcf()
    fig.set_size_inches(9.97,(12.44/4)*(len(mod_names)+1)) #w hh
    plt.tight_layout()
     
    if fpath_out is not None:
        plt.savefig(fpath_out, dpi=150, bbox_inches='tight')

def plot_temporal_sd(ds_d, mod_names, mod_longnames, months=None, fpath_out=None):
        
    err_ls = [validate_temporal_sd(ds_d[mod_name], ds_d.obs, months=months)
              for mod_name in mod_names]
        
    ds_metrics = xr.concat([a_err[['mod','err','pct_err']] for a_err in err_ls],
                           pd.Index(mod_longnames, name='mod_name'))
    ds_metrics = xr.merge([err_ls[0].obs,ds_metrics])
        
    ds_metrics.obs.attrs['longname'] = "Temporal Standard Deviation"
    ds_metrics.mod.attrs['longname'] = "Temporal Standard Deviation"
    ds_metrics.pct_err.attrs['longname'] = 'Percent Error (%)'
    ds_metrics.err.attrs['longname'] = 'Error (mm)'

    grid = plot_validation_grid_maps(ds_metrics, ['pct_err','err'], esd.cfg.bbox)
    for i in np.arange((len(mod_names)*3)+1,step=3):# [0,3,6,9]:
        cticks = grid.cbar_axes[i].get_xticklabels()
        cticks = np.array(cticks)
        cticks[np.arange(1,cticks.size,step=2)] = ''
        grid.cbar_axes[i].set_xticklabels(cticks)
    
    fig = plt.gcf()
    fig.set_size_inches(9.97,(12.44/4)*(len(mod_names)+1)) #w h
    plt.tight_layout()
     
    if fpath_out is not None:
        plt.savefig(fpath_out, dpi=150, bbox_inches='tight')

def plot_heidke_skill_score(ds_d, mod_names, mod_longnames, wet_thres, months=None, fpath_out=None):
    
    hss_ls = [validate_hss(ds_d[mod_name], ds_d.obs, wet_thres, months)
              for mod_name in mod_names]
            
    lon = hss_ls[0].lon
    lat = hss_ls[0].lat
 
    buf = .5
    bbox = esd.cfg.bbox
    m = Basemap(resolution='i',projection='cyl', llcrnrlat=bbox[1]-buf,
                area_thresh=100000, urcrnrlat=bbox[-1]+buf,
                llcrnrlon=bbox[0]-buf, urcrnrlon=bbox[2]+buf)
     
    x_map, y_map = m(*np.meshgrid(lon,lat))
     
    axes_pad = .4
    cfig = plt.gcf()
    
    nrow = int(np.ceil(len(mod_names)/3.0))
    grid = ImageGrid(cfig,111,nrows_ncols=(nrow, 3),
                     axes_pad=axes_pad, cbar_mode="each", cbar_pad=.1, cbar_location="bottom",
                     cbar_size='4%')
    
    levels = get_levels(np.concatenate([a_correl.values.ravel() for a_correl in hss_ls]), diverging=False)
                  
    #TODO: make dynamic     
    lon_txt = 99.8
    lat_txt = 25.75
    x_txt,y_txt = m(lon_txt, lat_txt)
    
    def contour_da(da, i, title):
    
        m.ax = grid[i]
        cntr = m.contourf(x_map,y_map,da.values,levels=levels, cmap='viridis', extend='both')
        cbar = plt.colorbar(cntr, cax = grid.cbar_axes[i],orientation='horizontal',extend='both')
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_xlabel('r',fontsize=8) 
        m.drawcountries()
        m.drawcoastlines()
        m.ax.text(x_txt,y_txt,'Mean: %.2f'%(da.mean(),),fontsize=8)
        m.ax.text(x_txt+2,y_txt,title,fontsize=8)
        #grid[i].set_title(title, fontsize=8)
    
    i = 0
    
    for a_correl,a_modname in zip(hss_ls, mod_longnames):
        contour_da(a_correl, i,a_modname)
        i+=1
        
    for j in np.arange(i,len(grid)):
        m.ax = grid[j]
        m.ax.axis('off')
        grid.cbar_axes[j].axis('off')
        
    
    for i in np.arange(len(mod_names)):
        cticks = grid.cbar_axes[i].get_xticklabels()
        cticks = np.array(cticks)
        cticks[np.arange(1,cticks.size,step=2)] = ''
        grid.cbar_axes[i].set_xticklabels(cticks)
    
    
    plt.suptitle('Heidke Skill Score (%.2f mm wet threshold)'%wet_thres)
    
    fig = plt.gcf()
    fig.set_size_inches(16.350,(11.52/2)*(nrow)) #w h
    plt.tight_layout()
     
    if fpath_out is not None:
        plt.savefig(fpath_out, dpi=150, bbox_inches='tight')
    
    return grid

def plot_spatial_sd_mthly(ds_d, mod_names, mod_longnames, fpath_out=None):
        
    sd_ls = [validate_spatial_sd_monthly(ds_d[mod_name])
              for mod_name in mod_names]
    sd_ls.append(validate_spatial_sd_monthly(ds_d.obs))
    
    sd_df = pd.concat(sd_ls, axis=1)
    sd_df.columns = mod_longnames + ['Observed']

    with plt.style.context(('seaborn-whitegrid')):
        sns.set_context('poster')
        sd_df.plot(color=['#1b9e77','#d95f02','#7570b3'])
        plt.ylabel("Average Anomaly Spatial Standard Deviation")
        plt.legend()
        plt.tight_layout()
    
    if fpath_out is not None:
        plt.savefig(fpath_out, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    
#     downscale_start_year = '1961'
#     downscale_end_year = '1977'
    
    downscale_start_year = '1991'
    downscale_end_year = '2007'
    
    fpath_ds_d = os.path.join(esd.cfg.data_root,'downscaling',
                              'downscale_tests_%s_%s.nc'%(downscale_start_year,
                                                          downscale_end_year))
    path_out = os.path.join(esd.cfg.data_root, 'downscaling', 'figures',
                            'downscale_test_%s_%s'%(downscale_start_year, downscale_end_year),'cornell_talk')
    mkdir_p(path_out)
    
    ds_d = xr.open_dataset(fpath_ds_d).load()
    mask_null = xr.concat([ds_d[vname].isnull().any(dim='time')
                           for vname in ds_d.data_vars.keys()],'d').any(dim='d')
    for vname in ds_d.data_vars.keys():
        ds_d[vname].values[:,mask_null.values] = np.nan
    
#     mod_names = ['mod_d_cg','mod_d','mod_d_anoms','mod_d_sgrid','mod_d_anoms_sgrid']
#     mod_longnames = ['Coarsened\nAphrodite', 'Analog Using\nAbsolute',
#                      'Analog Using\nAnomalies','Analog Using\nAbsolute (Interp Scale)',
#                      'Analog Using\nAnomalies (Interp Scale)']
    
#     mod_names = ['mod_d_cg','mod_d','mod_d_anoms']
#     mod_longnames = ['Coarsened\nAphrodite', 'Analog Using\nAbsolute','Analog Using\nAnomalies']
    
    mod_names = ['mod_d_cg','mod_d_anoms']
    mod_longnames = ['Coarsened\nAphrodite', 'CASAA']
    
    import seaborn as sns
    sns.set_context('poster')
    sns.set_style('whitegrid')
    # Average total precipitation
    fpath_out = os.path.join(path_out,'total_prcp_ann.png')
    plot_avg_total_prcp(ds_d,mod_names,mod_longnames,months=None,fpath_out=fpath_out)
    plt.clf()
    fpath_out = os.path.join(path_out,'total_prcp_dryseason.png')
    plot_avg_total_prcp(ds_d,mod_names,mod_longnames,months=[12,1,2],fpath_out=fpath_out)
    plt.clf()
    fpath_out = os.path.join(path_out,'total_prcp_monsoon.png')
    plot_avg_total_prcp(ds_d,mod_names,mod_longnames,months=[6,7,8,9],fpath_out=fpath_out)
    plt.clf()
    
    # Average wet day fraction
    # 0 mm threshold for wet day
    wet_thres=0
    fpath_out = os.path.join(path_out,'wet_day_frac_%.2fthres_ann.png'%wet_thres)
    plot_avg_wetday_frac(ds_d, mod_names, mod_longnames, wet_thres=wet_thres, months=None, fpath_out=fpath_out)
    plt.clf()
    fpath_out = os.path.join(path_out,'wet_day_frac_%.2fthres_dryseason.png'%wet_thres)
    plot_avg_wetday_frac(ds_d, mod_names, mod_longnames, wet_thres=wet_thres, months=[12,1,2], fpath_out=fpath_out)
    plt.clf()
    fpath_out = os.path.join(path_out,'wet_day_frac_%.2fthres_monsoon.png'%wet_thres)
    plot_avg_wetday_frac(ds_d, mod_names, mod_longnames, wet_thres=wet_thres, months=[6,7,8,9], fpath_out=fpath_out)
    plt.clf()
    
    wet_thres=0.1
    # 0.1 mm threshold for wet day
    fpath_out = os.path.join(path_out,'wet_day_frac_%.2fthres_ann.png'%wet_thres)
    plot_avg_wetday_frac(ds_d, mod_names, mod_longnames, wet_thres=wet_thres, months=None, fpath_out=fpath_out)
    plt.clf()
    fpath_out = os.path.join(path_out,'wet_day_frac_%.2fthres_dryseason.png'%wet_thres)
    plot_avg_wetday_frac(ds_d, mod_names, mod_longnames, wet_thres=wet_thres, months=[12,1,2], fpath_out=fpath_out)
    plt.clf()
    fpath_out = os.path.join(path_out,'wet_day_frac_%.2fthres_monsoon.png'%wet_thres)
    plot_avg_wetday_frac(ds_d, mod_names, mod_longnames, wet_thres=wet_thres, months=[6,7,8,9], fpath_out=fpath_out)
    plt.clf()
    
    # Quantiles
    q=99.9
    fpath_out = os.path.join(path_out,'percentile%.2f_ann.png'%q)
    plot_quantile(ds_d, mod_names, mod_longnames, q=q, months=None, fpath_out=fpath_out)
    plt.clf()
    
    fpath_out = os.path.join(path_out,'percentile%.2f_dryseason.png'%q)
    plot_quantile(ds_d, mod_names, mod_longnames, q=q, months=[12,1,2], fpath_out=fpath_out)
    plt.clf()
    
    fpath_out = os.path.join(path_out,'percentile%.2f_monsoon.png'%q)
    plot_quantile(ds_d, mod_names, mod_longnames, q=q, months=[6,7,8,9], fpath_out=fpath_out)
    plt.clf()
        
    # Correlation
    # Spearman
    rfunc = lambda x,y: spearmanr(x,y)[0]
    #rfunc = lambda x,y: pearsonr(x,y)[0]
    fpath_out = os.path.join(path_out, 'correl_spearman_all_months.png')
    plot_correlation(ds_d, mod_names, mod_longnames, rfunc,
                     "Spearman Correlation: All Months", months=None, fpath_out=fpath_out)
    plt.clf()

    fpath_out = os.path.join(path_out, 'correl_spearman_monsoon.png')
    plot_correlation(ds_d, mod_names, mod_longnames, rfunc,
                     "Spearman Correlation: Monsoon Season", months=[6,7,8,9], fpath_out=fpath_out)
    plt.clf()

    fpath_out = os.path.join(path_out, 'correl_spearman_dryseason.png')
    plot_correlation(ds_d, mod_names, mod_longnames, rfunc,
                     "Spearman Correlation: Dry Season", months=[12,1,2], fpath_out=fpath_out)
    plt.clf()
        
    # Spatiotemporal Coherence of daily anomalies
    # Annual
    df_stcorrel = pd.read_csv(os.path.join(esd.cfg.data_root,'downscaling', 'st_correl',
                                           'stcorrel_anomalies_ann_%s_%s.csv'%(downscale_start_year,
                                                                     downscale_end_year)),index_col=0)
    df_stcorrel = df_stcorrel.rename(columns={'Analog Using\nAnomalies':'CASAA'})
    
    fpath_out = os.path.join(path_out, 'stcorrelogram_ann.png')
    plot_st_correlogram(df_stcorrel, mod_longnames, fpath_out)
    plt.clf()
    # Dry Season
    df_stcorrel = pd.read_csv(os.path.join(esd.cfg.data_root,'downscaling', 'st_correl',
                                           'stcorrel_anomalies_dryseason_%s_%s.csv'%(downscale_start_year,
                                                                           downscale_end_year)),index_col=0)
    df_stcorrel = df_stcorrel.rename(columns={'Analog Using\nAnomalies':'CASAA'})
    fpath_out = os.path.join(path_out, 'stcorrelogram_dryseason.png')
    plot_st_correlogram(df_stcorrel, mod_longnames, fpath_out)
    plt.clf()
    # Monsoon Season
    df_stcorrel = pd.read_csv(os.path.join(esd.cfg.data_root,'downscaling', 'st_correl',
                                           'stcorrel_anomalies_monsoonseason_%s_%s.csv'%(downscale_start_year,
                                                                               downscale_end_year)),index_col=0)
    fpath_out = os.path.join(path_out, 'stcorrelogram_monsoon.png')
    plot_st_correlogram(df_stcorrel, mod_longnames, fpath_out)
    plt.clf()
        
    # Wet/dry spell metrics    
    # Mean wet spell length
    wetspell_func = np.mean
    wet_thres = .1
    fpath_out = os.path.join(path_out, 'wetspell_mean_ann.png')
    plot_wetspell(ds_d, mod_names, mod_longnames, metric_func=wetspell_func,
                  title='Mean Wet Spell Length (days)', wet_thres=wet_thres, months=None, fpath_out=fpath_out)
    plt.clf()
    fpath_out = os.path.join(path_out, 'wetspell_mean_monsoon.png')
    plot_wetspell(ds_d, mod_names, mod_longnames, metric_func=wetspell_func,
                  title='Mean Wet Spell Length (days)', wet_thres=wet_thres, months=[6,7,8,9], fpath_out=fpath_out)
    plt.clf()
    fpath_out = os.path.join(path_out, 'wetspell_mean_dryseason.png')
    plot_wetspell(ds_d, mod_names, mod_longnames, metric_func=wetspell_func,
                  title='Mean Wet Spell Length (days)', wet_thres=wet_thres, months=[12,1,2], fpath_out=fpath_out)
    plt.clf()
    
    # 95th percentile wet spell length
    wetspell_func = lambda a: np.percentile(a, 95)
    wet_thres = .1
    fpath_out = os.path.join(path_out, 'wetspell_95p_ann.png')
    plot_wetspell(ds_d, mod_names, mod_longnames, metric_func=wetspell_func,
                  title='95th Ptile Wet Spell Length (days)', wet_thres=wet_thres, months=None, fpath_out=fpath_out)
    plt.clf()
    fpath_out = os.path.join(path_out, 'wetspell_95p_monsoon.png')
    plot_wetspell(ds_d, mod_names, mod_longnames, metric_func=wetspell_func,
                  title='95th Ptile Wet Spell Length (days)', wet_thres=wet_thres, months=[6,7,8,9], fpath_out=fpath_out)
    plt.clf()
    fpath_out = os.path.join(path_out, 'wetspell_95p_dryseason.png')
    plot_wetspell(ds_d, mod_names, mod_longnames, metric_func=wetspell_func,
                  title='95th Ptile Wet Spell Length (days)', wet_thres=wet_thres, months=[12,1,2], fpath_out=fpath_out)
    plt.clf()
    
    # Temporal Standard Deviation
    fpath_out = os.path.join(path_out, 'temporalsd_ann.png')
    plot_temporal_sd(ds_d, mod_names, mod_longnames, months=None, fpath_out=fpath_out)
    plt.clf()
    
    fpath_out = os.path.join(path_out, 'temporalsd_dryseason.png')
    plot_temporal_sd(ds_d, mod_names, mod_longnames, months=[12,1,2], fpath_out=fpath_out)
    plt.clf()
    
    fpath_out = os.path.join(path_out, 'temporalsd_monsoon.png')
    plot_temporal_sd(ds_d, mod_names, mod_longnames, months=[6,7,8,9], fpath_out=fpath_out)
    plt.clf()
    
    # Heidke Skill Score
    # Run for 0 and 0.1 mm wet day thresholds
    wet_thresholds = [0, 0.1]
    
    for a_wet_thres in wet_thresholds:
        
        fpath_out = os.path.join(path_out, 'hss_%.2fthres_ann.png'%a_wet_thres)
        plot_heidke_skill_score(ds_d, mod_names, mod_longnames, wet_thres=a_wet_thres, months=None, fpath_out=fpath_out)
        plt.clf()
    
        fpath_out = os.path.join(path_out, 'hss_%.2fthres_monsoon.png'%a_wet_thres)
        plot_heidke_skill_score(ds_d, mod_names, mod_longnames, wet_thres=a_wet_thres, months=[6,7,8,9], fpath_out=fpath_out)
        plt.clf()
    
        fpath_out = os.path.join(path_out, 'hss_%.2fthres_dryseason.png'%a_wet_thres)
        plot_heidke_skill_score(ds_d, mod_names, mod_longnames, wet_thres=a_wet_thres, months=[12,1,2], fpath_out=fpath_out)
        plt.clf()
        
    # Spatial variability
    fpath_out = os.path.join(path_out, 'spatial_sd_anom.png')
    ds_d_anoms = []
    for vname in mod_names+['obs']:#ds_d_anoms.data_vars.keys():
        ds_d_anoms.append(to_anomalies(ds_d[vname])[0])
    ds_d_anoms = xr.merge(ds_d_anoms)
    plot_spatial_sd_mthly(ds_d_anoms, mod_names, mod_longnames, fpath_out=fpath_out)
        

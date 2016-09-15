'''
Utility functions for plotting
'''

from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from mpl_toolkits.basemap import Basemap
import math
import matplotlib.pyplot as plt
import numpy as np

def precision_and_scale(x):
    '''
    http://stackoverflow.com/questions/3018758/determine-precision-and-scale-of-particular-number-in-python
    '''
    
    max_digits = 14
    int_part = int(abs(x))
    magnitude = 1 if int_part == 0 else int(math.log10(int_part)) + 1
    if magnitude >= max_digits:
        return (magnitude, 0)
    frac_part = abs(x) - int_part
    multiplier = 10 ** (max_digits - magnitude)
    frac_digits = multiplier + int(multiplier * frac_part + 0.5)
    while frac_digits % 10 == 0:
        frac_digits /= 10
    scale = int(math.log10(frac_digits))
    return (magnitude + scale, scale)


def get_levels(a, diverging):
    
    val_pctl2 = np.nanpercentile(a,2)
    val_pctl98 = np.nanpercentile(a,98)
    
    if diverging:
    
        val_limit = np.max(np.abs([val_pctl2,val_pctl98]))
        # Round to 1 significant digits
        val_limit = float("%.2g"%val_limit)
        val_start = -val_limit
        val_end = val_limit
        
    else:
        
        val_start = float("%.2g"%val_pctl2)
        val_end = float("%.2g"%val_pctl98)
        
    val_range = val_end - val_start
    
    val_step = float("%.1g"%((val_range)/50.0))
    
    p,s = precision_and_scale(val_step)
     
    if s == 0:
        val_step = 10**(p-1)
    else:
        val_step = 10**-s
    
    val_n = int(np.round((val_range/val_step) + 1))
    
    levels = np.linspace(val_start, val_end, num=val_n)
    
    return levels

def plot_validation_grid_maps(ds, err_metrics, bbox):
     
    lon = ds.lon
    lat = ds.lat
 
    buf = .5
    m = Basemap(resolution='i',projection='cyl', llcrnrlat=bbox[1]-buf,
                area_thresh=100000, urcrnrlat=bbox[-1]+buf,
                llcrnrlon=bbox[0]-buf, urcrnrlon=bbox[2]+buf)
     
    x_map, y_map = m(*np.meshgrid(lon,lat))
     
    axes_pad = .2
    cfig = plt.gcf()
    grid = ImageGrid(cfig,111,nrows_ncols=(1 + ds.mod_name.size, 1 + len(err_metrics)),
                     axes_pad=axes_pad, cbar_mode="each", cbar_pad=.1, cbar_location="bottom",
                     cbar_size='4%')
     
    def contour_a(a, diverging=True, levels=None):
         
        if levels is None:
            levels = get_levels(a, diverging)
             
        if diverging:
            cmap = 'RdBu_r'
        else:
            cmap = 'viridis'
         
        cntr = m.contourf(x_map,y_map,a,levels=levels, cmap=cmap, extend='both')
        return cntr
         
    levels_vals = get_levels(np.concatenate([ds.obs.values.ravel(), ds.mod.values.ravel()]), diverging=False)
    levels_err = {err_name:get_levels(ds[err_name].values.ravel(), diverging=True) for err_name in err_metrics}
    
    #TODO: make dynamic     
    lon_txt = 99.8
    lat_txt = 25.75
    x_txt,y_txt = m(lon_txt, lat_txt)
     
    def plot_a_model(mod_name, i):
         
        m.ax = grid[i]
        cntr = contour_a(ds.mod.loc[mod_name].values,diverging=False,levels=levels_vals)
        cbar = plt.colorbar(cntr, cax = grid.cbar_axes[i],orientation='horizontal',extend='both')
        cbar.ax.tick_params(labelsize=8) 
        m.drawcountries()
        m.drawcoastlines()
        m.ax.text(x_txt,y_txt,'Mean: %.2f'%(ds.mod.loc[mod_name].mean(),),fontsize=8)
        
        for j,err_name in enumerate(err_metrics):
            
            m.ax = grid[i+j+1]
            cntr = contour_a(ds[err_name].loc[mod_name].values, levels=levels_err[err_name])
            cbar = plt.colorbar(cntr, cax = grid.cbar_axes[i+j+1],orientation='horizontal',extend='both')
            cbar.ax.tick_params(labelsize=8) 
            m.drawcountries()
            m.drawcoastlines()
            m.ax.text(x_txt,y_txt,'MAE: %.2f  Mean: %.2f'%(np.abs(ds[err_name].loc[mod_name]).mean(),
                                                           ds[err_name].loc[mod_name].mean()),fontsize=8)
                 
    m.ax = grid[0]
    cntr = contour_a(ds.obs.values,diverging=False,levels=levels_vals)
    cbar = plt.colorbar(cntr, cax = grid.cbar_axes[0],orientation='horizontal',extend='both')
    cbar.ax.tick_params(labelsize=8)
    x,y = m(lon_txt, lat_txt)
    m.ax.text(x,y,'Mean: %.2f'%(ds.obs.mean(),),fontsize=8) 
    m.drawcountries()
    m.drawcoastlines()
    
    for j,err_name in enumerate(err_metrics):
    
        m.ax = grid[j+1]
        m.ax.axis('off')
        grid.cbar_axes[j+1].axis('off')
             
    nplot = len(err_metrics) + 1
    
    for mod_name in ds.mod_name.values:
        
        plot_a_model(mod_name, nplot)
        nplot = nplot + len(err_metrics) + 1
     
    # Set titles
    grid[0].set_title(ds.obs.longname, fontsize=8)
    nplot = len(err_metrics) + 2
    for j,err_name in enumerate(err_metrics):
        grid[nplot+j].set_title(ds[err_name].longname, fontsize=8)
     
    # Set y labels
    grid[0].set_ylabel('Observed',fontsize=8)
    nplot = len(err_metrics) + 1
    for mod_name in ds.mod_name.values:
        grid[nplot].set_ylabel(mod_name,fontsize=8)
        nplot = nplot + len(err_metrics) + 1
    
    return grid

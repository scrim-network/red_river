'''
Script for generating joint scatter plots of GCM precipitation and temperature
trends.
'''

import esd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

def plot_season_trends(season, time_period):
        
    pr_trends = df_trends.query("elem=='pr' & season=='%s' & time_period=='%s'"%(season,time_period))
    tas_trends = df_trends.query("elem=='tas' & season=='%s' & time_period=='%s'"%(season,time_period))
    
    pr_trends = pr_trends.set_index('model_run').trend
    tas_trends = tas_trends.set_index('model_run').trend
        
    run_names = np.intersect1d(pr_trends.index.values, tas_trends.index.values)
    
    pr_trends = pr_trends.loc[run_names]
    tas_trends = tas_trends.loc[run_names]
    
    tas_trends.name = r'Tair ($^\circ$C decade$^{-1}$)'
    pr_trends.name = r'Prcp (% decade$^{-1}$)'
    
    df = pd.DataFrame({tas_trends.name:tas_trends, pr_trends.name:pr_trends})
    df['model_name'] = df.index.values
    df['rcp'] = df['model_name'].str.split('_',expand=True)[1].values
    pt_colors = {'rcp26':'#377eb8','rcp45':'#4daf4a','rcp60':'#984ea3','rcp85':'#e41a1c'}
               
    g = sns.jointplot(tas_trends.name,pr_trends.name,data=df,kind='kde',
                      stat_func=None,shade_lowest=False,legend=True,color='k')
    
    pt_ls = []
    
    for rcp in np.unique(df.rcp.values):
        
        df_rcp = df[df.rcp==rcp]
        pt_ls.append(g.ax_joint.scatter(df_rcp[tas_trends.name].values,
                                        df_rcp[pr_trends.name].values, c=pt_colors[rcp]))
    
    leg = g.ax_joint.legend(pt_ls,np.unique(df.rcp.values),frameon=True)
    leg.get_frame().set_edgecolor('k')
    
    g.ax_joint.set_ylim((-5,8))
    g.ax_joint.set_xlim((-.1,.8))
        
    plt.title("\n".join([season,time_period]))
    g.ax_joint.set_yticks(np.arange(-5,9))
    
    plt.sca(g.ax_joint)
    plt.hlines(0, g.ax_joint.get_xlim()[0], g.ax_joint.get_xlim()[1],lw=.5)
    plt.vlines(0, g.ax_joint.get_ylim()[0], g.ax_joint.get_ylim()[1],lw=.5)

if __name__ == '__main__':
    
    # Trends for CMIP5 realizations have NOT been biased corrected and downscaled
#     fpath_trend_csv = os.path.join(esd.cfg.path_cmip5_trends, 'red_river_trends.csv')
#     path_out_figs = os.path.join(esd.cfg.path_cmip5_trends, 'figures')
    
    # Trends for CMIP5 realizations have been biased corrected and downscaled
    fpath_trend_csv = os.path.join(esd.cfg.path_cmip5_trends, 'downscaled', 'red_river_trends.csv')
    path_out_figs = os.path.join(esd.cfg.path_cmip5_trends, 'downscaled', 'figures')
    
    df_trends = pd.read_csv(fpath_trend_csv)

    #sns.set_context('talk')
    plt.clf()
    season = 'JJAS' #'ANNUAL' 'JJAS'
    time_period = '2006_2050'#'2006_2099' '2006_2050'
    plot_season_trends(season, time_period)
    plt.tight_layout()
    plt.savefig(os.path.join(path_out_figs, 'scatter_trends_%s_%s.png'%(season,time_period)),
                dpi=300, bbox_inches='tight')

    plt.clf()
    season = 'JJAS' #'ANNUAL' 'JJAS'
    time_period = '2006_2099'#'2006_2099' '2006_2050'
    plot_season_trends(season, time_period)
    plt.tight_layout()
    plt.savefig(os.path.join(path_out_figs, 'scatter_trends_%s_%s.png'%(season,time_period)),
                dpi=300, bbox_inches='tight')
    
    plt.clf()
    season = 'ANNUAL' #'ANNUAL' 'JJAS'
    time_period = '2006_2050'#'2006_2099' '2006_2050'
    plot_season_trends(season, time_period)
    plt.tight_layout()
    plt.savefig(os.path.join(path_out_figs, 'scatter_trends_%s_%s.png'%(season,time_period)),
                dpi=300, bbox_inches='tight')

    plt.clf()
    season = 'ANNUAL' #'ANNUAL' 'JJAS'
    time_period = '2006_2099'#'2006_2099' '2006_2050'
    plot_season_trends(season, time_period)
    plt.tight_layout()
    plt.savefig(os.path.join(path_out_figs, 'scatter_trends_%s_%s.png'%(season,time_period)),
                dpi=300, bbox_inches='tight')
        
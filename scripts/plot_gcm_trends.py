import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def plot_mthly_mean_trends():
    
    pr_trends = pd.read_csv('/storage/home/jwo118/group_store/red_river/cmip5_trends/pr_2006_2050.csv',index_col='model_run')
    tas_trends = pd.read_csv('/storage/home/jwo118/group_store/red_river/cmip5_trends/tas_2006_2050.csv',index_col='model_run')
    
    pr_trends = pd.read_csv('/storage/home/jwo118/group_store/red_river/cmip5_trends/pr_2006_2099.csv',index_col='model_run')
    tas_trends = pd.read_csv('/storage/home/jwo118/group_store/red_river/cmip5_trends/tas_2006_2099.csv',index_col='model_run')
    
    #pr_trends = pr_trends * 9.4
    #tas_trends = tas_trends * 9.4
    
    run_names = np.intersect1d(pr_trends.index.values, tas_trends.index.values)
    
    pr_trends = pr_trends.loc[run_names]
    tas_trends = tas_trends.loc[run_names]
    
    tas_trends_mean = tas_trends.mean(axis=1)
    tas_trends_mean.name = r'Tair ($^\circ$C decade$^{-1}$)'
    
    pr_trends_mean = pr_trends.mean(axis=1)
    pr_trends_mean.name = r'Prcp (% decade$^{-1}$)'
    
    df = pd.DataFrame({tas_trends_mean.name:tas_trends_mean,pr_trends_mean.name:pr_trends_mean})
    df['model_name'] = df.index.values
    df['rcp'] = df['model_name'].str.split('_',expand=True)[1].values
    pt_colors = {'rcp26':'#377eb8','rcp45':'#4daf4a','rcp60':'#984ea3','rcp85':'#e41a1c'}
               
    g = sns.jointplot(tas_trends_mean.name,pr_trends_mean.name,data=df,kind='kde',stat_func=None,shade_lowest=False,legend=True,color='k')
    
    pt_ls = []
    
    for rcp in np.unique(df.rcp.values):
        
        df_rcp = df[df.rcp==rcp]
        pt_ls.append(g.ax_joint.scatter(df_rcp[tas_trends_mean.name].values, df_rcp[pr_trends_mean.name].values, c=pt_colors[rcp]))
    
    leg = g.ax_joint.legend(pt_ls,np.unique(df.rcp.values),frameon=True)
    leg.get_frame().set_edgecolor('k')
    
    #s = g.plot_joint(plt.scatter,c='w',linewidth=1)
    #s = g.plot_joint(plt.scatter,c=[pt_colors[rcp] for rcp in df['rcp'].values])#,linewidth=1)
    g.ax_joint.set_ylim((-10,10))
    g.ax_joint.set_xlim((-.1,.8))

def plot_season_trends():
    
    df_trends = pd.read_csv('/storage/home/jwo118/group_store/red_river/cmip5_trends/all_trends/red_river_projections.csv')
    
    season = 'JJAS' #'ANNUAL' 'JJAS'
    time_period = '2006_2050'#'2006_2099' '2006_2050'
    
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
               
    g = sns.jointplot(tas_trends.name,pr_trends.name,data=df,kind='kde',stat_func=None,shade_lowest=False,legend=True,color='k')
    
    pt_ls = []
    
    for rcp in np.unique(df.rcp.values):
        
        df_rcp = df[df.rcp==rcp]
        pt_ls.append(g.ax_joint.scatter(df_rcp[tas_trends.name].values, df_rcp[pr_trends.name].values, c=pt_colors[rcp]))
    
    leg = g.ax_joint.legend(pt_ls,np.unique(df.rcp.values),frameon=True)
    leg.get_frame().set_edgecolor('k')
    
    #s = g.plot_joint(plt.scatter,c='w',linewidth=1)
    #s = g.plot_joint(plt.scatter,c=[pt_colors[rcp] for rcp in df['rcp'].values])#,linewidth=1)
    g.ax_joint.set_ylim((-5,8))
    g.ax_joint.set_xlim((-.1,.8))
        
    plt.title("\n".join([season,time_period]))
    g.ax_joint.set_yticks(np.arange(-5,9))
    
    plt.sca(g.ax_joint)
    plt.hlines(0, g.ax_joint.get_xlim()[0], g.ax_joint.get_xlim()[1],lw=.5)#,zorder=-20)
    plt.vlines(0, g.ax_joint.get_ylim()[0], g.ax_joint.get_ylim()[1],lw=.5)#zorder=-20)

if __name__ == '__main__':
    
    anoms_tair = pd.read_csv('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_tas_2006_2099_mths4-5.csv',
                             index_col='time')
    rcp = np.array([r.split('_')[1] for r in anoms_tair.columns.values])
    urcp = np.unique(rcp)
    rcp_means_tair = pd.concat([anoms_tair.loc[:,rcp==r].mean(axis=1) for r in urcp],axis=1)
    rcp_means_tair.columns = urcp
    
    anoms_prcp = pd.read_csv('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_pr_2006_2099_mths4-5.csv',
                             index_col='time')
    rcp = np.array([r.split('_')[1] for r in anoms_prcp.columns.values])
    urcp = np.unique(rcp)
    rcp_means_prcp = pd.concat([anoms_prcp.loc[:,rcp==r].mean(axis=1) for r in urcp],axis=1)
    rcp_means_prcp.columns = urcp
    
    rcp_means_tair.plot()
    plt.ylabel("Anomaly (degC)")
    
    rcp_means_prcp.rolling(10, center=True).mean().plot()
    plt.ylabel("Fraction of Baseline Normal")
    
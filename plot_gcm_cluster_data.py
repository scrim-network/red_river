import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

anom_fpaths = [('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_tas_2006_2099_ann.csv','tair_ann'),
               ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_tas_2006_2099_mths10-11.csv','tair_mths10-11'),
               ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_tas_2006_2099_mths12-1-2.csv','tair_mths12-1-2'),
               ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_tas_2006_2099_mths4-5.csv','tair_mths4-5'),
               ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_tas_2006_2099_mths6-7-8-9.csv','tair_mths6-7-8-9'),
               ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_pr_2006_2099_ann.csv','prcp_ann'),
               ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_pr_2006_2099_mths10-11.csv','prcp_mths10-11'),
               ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_pr_2006_2099_mths12-1-2.csv','prcp_mths12-1-2'),
               ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_pr_2006_2099_mths4-5.csv','prcp_mths4-5'),
               ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_pr_2006_2099_mths6-7-8-9.csv','prcp_mths6-7-8-9')]

def line_rcp():
        
    anoms_all = []
    
    for fpath,vname in anom_fpaths[5:]:#anom_fpaths[5:]:
        
        print vname
        
        anoms = pd.read_csv(fpath, index_col='time')
        anoms.index = pd.DatetimeIndex(anoms.index)
        anoms = anoms.rolling(10, center=True).mean()
        
        rcp = np.array([r.split('_')[1] for r in anoms.columns.values])
        urcp = np.unique(rcp)
        rcp_means = pd.concat([anoms.loc[:,rcp==r].mean(axis=1) for r in urcp],axis=1)
        rcp_means.columns = urcp
            
        rcp_means = rcp_means.stack()
        rcp_means = rcp_means.reset_index()
        rcp_means.columns = ['time','rcp','anomaly']
        rcp_means['vname'] = vname
        
        anoms_all.append(rcp_means)
        
    anoms_all = pd.concat(anoms_all,axis=0,ignore_index=True)
    
    # Tair
    anoms_all = anoms_all.rename(columns={'anomaly':'anomaly (degC)','vname':'time window'})
    anoms_all = anoms_all.replace('tair_ann','annual')
    anoms_all = anoms_all.replace('tair_mths10-11','Oct,Nov')
    anoms_all = anoms_all.replace('tair_mths12-1-2','Dec,Jan,Feb')
    anoms_all = anoms_all.replace('tair_mths4-5','Apr,May')
    anoms_all = anoms_all.replace('tair_mths6-7-8-9','Jun,Jul,Aug,Sep')
    
    g = sns.FacetGrid(anoms_all, col="time window", hue="rcp",col_wrap=2,sharey=True)
    g = (g.map(plt.plot, "time", "anomaly (degC)"))
    plt.legend(bbox_to_anchor=(1.75, 1))
    
    # Prcp
    anoms_all = anoms_all.rename(columns={'anomaly':'anomaly (fraction)','vname':'time window'})
    anoms_all = anoms_all.replace('prcp_ann','annual')
    anoms_all = anoms_all.replace('prcp_mths10-11','Oct,Nov')
    anoms_all = anoms_all.replace('prcp_mths12-1-2','Dec,Jan,Feb')
    anoms_all = anoms_all.replace('prcp_mths4-5','Apr,May')
    anoms_all = anoms_all.replace('prcp_mths6-7-8-9','Jun,Jul,Aug,Sep')
    
    g = sns.FacetGrid(anoms_all, col="time window", hue="rcp",col_wrap=2,sharey=False)
    g = (g.map(plt.plot, "time", "anomaly (fraction)"))
    plt.legend(bbox_to_anchor=(1.75, 1))
    
    for ax in g.axes:
        ax.axhline(1,color='k',lw=.5)
    
def line_gcm_cluster():
    
    
    clusters = pd.read_csv('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/cluster_nums.csv')
    clusters.columns = ['model_name','k.cluster']
    clusters = clusters.set_index('model_name')
    
    anoms_all = []
    
    for fpath,vname in anom_fpaths:#anom_fpaths[5:]:
        
        print vname
        
        anoms = pd.read_csv(fpath, index_col='time')
        anoms.index = pd.DatetimeIndex(anoms.index)
        anoms = anoms.rolling(10, center=True).mean()
        
        clusters_col = clusters.loc[anoms.columns,'k.cluster']
        cluster_means = pd.concat([anoms.loc[:,clusters_col==c].mean(axis=1) for c in np.arange(1,13)],axis=1)
        cluster_means.columns = ["cluster%.2d"%(i+1) for i in cluster_means.columns]
        
        cluster_means = cluster_means.stack()
        cluster_means = cluster_means.reset_index()
        cluster_means.columns = ['time','cluster','anomaly']
        cluster_means['vname'] = vname
        
        anoms_all.append(cluster_means)
        
    anoms_all = pd.concat(anoms_all,axis=0,ignore_index=True)
    
    # Tair
    anoms_all = anoms_all.rename(columns={'anomaly':'anomaly (degC)','vname':'time window'})
    anoms_all = anoms_all.replace('tair_ann','annual')
    anoms_all = anoms_all.replace('tair_mths10-11','Oct,Nov')
    anoms_all = anoms_all.replace('tair_mths12-1-2','Dec,Jan,Feb')
    anoms_all = anoms_all.replace('tair_mths4-5','Apr,May')
    anoms_all = anoms_all.replace('tair_mths6-7-8-9','Jun,Jul,Aug,Sep')
    
    g = sns.FacetGrid(anoms_all, col="time window", hue="cluster",col_wrap=2,palette=sns.color_palette("husl", 12),sharey=True)
    g = (g.map(plt.plot, "time", "anomaly (degC)"))
    plt.legend(bbox_to_anchor=(1.75, 1))
    
    # Prcp
    anoms_all = anoms_all.rename(columns={'anomaly':'anomaly (fraction)','vname':'time window'})
    anoms_all = anoms_all.replace('prcp_ann','annual')
    anoms_all = anoms_all.replace('prcp_mths10-11','Oct,Nov')
    anoms_all = anoms_all.replace('prcp_mths12-1-2','Dec,Jan,Feb')
    anoms_all = anoms_all.replace('prcp_mths4-5','Apr,May')
    anoms_all = anoms_all.replace('prcp_mths6-7-8-9','Jun,Jul,Aug,Sep')
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    g = sns.FacetGrid(anoms_all, col="time window", hue="cluster",col_wrap=2,palette=sns.color_palette("husl", 12),sharey=False)
    g = (g.map(plt.plot, "time", "anomaly (fraction)"))
    plt.legend(bbox_to_anchor=(1.75, 1))
    
    for ax in g.axes:
        ax.axhline(1,color='k',lw=.5)
    
    
    anoms = pd.read_csv(anom_fpaths[0][0], index_col='time')
    
    clusters_col = clusters.loc[anoms.columns,'k.cluster']
    
    cluster_means = pd.concat([anoms.loc[:,clusters_col==c].mean(axis=1) for c in np.arange(1,13)],axis=1)

def heatmap_gcm_rcp():
    
    anoms_all = []
    
    for fpath,vname in anom_fpaths:#anom_fpaths[5:]:
        
        print vname
        
        anoms = pd.read_csv(fpath, index_col='time')
        anoms.index = pd.DatetimeIndex(anoms.index)
        #anoms = (anoms - anoms.mean())/ anoms.std()
        anoms = anoms['2010':].resample('5AS').mean()
        anoms.index = anoms.index.year
        
        rcp = np.array([r.split('_')[1] for r in anoms.columns.values])
        urcp = np.unique(rcp)
        rcp_means = pd.concat([anoms.loc[:,rcp==r].mean(axis=1) for r in urcp],axis=1)
        rcp_means.columns = urcp
            
        rcp_means = rcp_means.stack()
        rcp_means = rcp_means.reset_index()
        rcp_means.columns = ['year','rcp','anomaly']
        rcp_means['vname'] = vname
        
        anoms_all.append(rcp_means)
                
    anoms_all = pd.concat(anoms_all,axis=0,ignore_index=True)
    
    anoms_all_tair = anoms_all[anoms_all.vname.str.startswith('tair')]
    anoms_all_prcp = anoms_all[anoms_all.vname.str.startswith('prcp')]
    
    anoms_all_tair = anoms_all_tair.replace('tair_ann','Annual')
    anoms_all_tair = anoms_all_tair.replace('tair_mths10-11','Oct,Nov')
    anoms_all_tair = anoms_all_tair.replace('tair_mths12-1-2','Dec,Jan,Feb')
    anoms_all_tair = anoms_all_tair.replace('tair_mths4-5','Apr,May')
    anoms_all_tair = anoms_all_tair.replace('tair_mths6-7-8-9','Jun,Jul,Aug,Sep')
    anoms_all_prcp = anoms_all_prcp.replace('prcp_ann','Annual')
    anoms_all_prcp = anoms_all_prcp.replace('prcp_mths10-11','Oct,Nov')
    anoms_all_prcp = anoms_all_prcp.replace('prcp_mths12-1-2','Dec,Jan,Feb')
    anoms_all_prcp = anoms_all_prcp.replace('prcp_mths4-5','Apr,May')
    anoms_all_prcp = anoms_all_prcp.replace('prcp_mths6-7-8-9','Jun,Jul,Aug,Sep')
    
    anoms_all_tair = anoms_all_tair.set_index(['year','rcp','vname'])
    anoms_all_tair = anoms_all_tair.unstack('vname')
    anoms_all_prcp = anoms_all_prcp.set_index(['year','rcp','vname'])
    anoms_all_prcp = anoms_all_prcp.unstack('vname')
    
    i = 1
    for cname in anoms_all_tair.index.levels[1]:
                
        anoms_tair = anoms_all_tair.xs(cname,level='rcp').transpose()
        anoms_tair.index = anoms_tair.index.droplevel(0)
        anoms_tair.index.name = None
    
        anoms_prcp = anoms_all_prcp.xs(cname,level='rcp').transpose()
        anoms_prcp.index = anoms_prcp.index.droplevel(0)
        anoms_prcp.index.name = None
        


        ax = plt.subplot(4,2,i)
        i+=1
        sns.heatmap(anoms_tair,vmin=0,vmax=5,xticklabels=True, yticklabels=True,annot=True,fmt='.1f',annot_kws={'fontsize':7})
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.xlabel('')
        
        if cname != 'rcp85':
            plt.gca().get_xaxis().set_ticks([])
        
        
        ax = plt.subplot(4,2,i)
        i+=1
        
        sns.heatmap(anoms_prcp, cmap='RdBu', vmin=.9,vmax=1.1,xticklabels=True, yticklabels=True,annot=True,fmt='.2f',annot_kws={'fontsize':7})
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.xlabel('')
        plt.ylabel(cname)
        plt.gca().get_yaxis().set_ticks([])
        
        if cname != 'rcp85':
            plt.gca().get_xaxis().set_ticks([])


def heatmap_gcm_cluster():

    clusters = pd.read_csv('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/cluster_nums.csv')
    clusters.columns = ['model_name','k.cluster']
    clusters = clusters.set_index('model_name')
    
    anoms_all = []
    
    for fpath,vname in anom_fpaths:#anom_fpaths[5:]:
        
        print vname
        
        anoms = pd.read_csv(fpath, index_col='time')
        anoms.index = pd.DatetimeIndex(anoms.index)
        #anoms = (anoms - anoms.mean())/ anoms.std()
        anoms = anoms['2010':].resample('5AS').mean()
        anoms.index = anoms.index.year
        clusters_col = clusters.loc[anoms.columns,'k.cluster']
        cluster_means = pd.concat([anoms.loc[:,clusters_col==c].mean(axis=1) for c in np.arange(1,13)],axis=1)
        cluster_means.columns = ["cluster%.2d"%(i+1) for i in cluster_means.columns]
        
        cluster_means = cluster_means.stack()
        cluster_means = cluster_means.reset_index()
        cluster_means.columns = ['year','cluster','anomaly']
        cluster_means['vname'] = vname
        
        anoms_all.append(cluster_means)
        
    anoms_all = pd.concat(anoms_all,axis=0,ignore_index=True)
    
    anoms_all_tair = anoms_all[anoms_all.vname.str.startswith('tair')]
    anoms_all_prcp = anoms_all[anoms_all.vname.str.startswith('prcp')]
    
    anoms_all_tair = anoms_all_tair.replace('tair_ann','Annual')
    anoms_all_tair = anoms_all_tair.replace('tair_mths10-11','Oct,Nov')
    anoms_all_tair = anoms_all_tair.replace('tair_mths12-1-2','Dec,Jan,Feb')
    anoms_all_tair = anoms_all_tair.replace('tair_mths4-5','Apr,May')
    anoms_all_tair = anoms_all_tair.replace('tair_mths6-7-8-9','Jun,Jul,Aug,Sep')
    anoms_all_prcp = anoms_all_prcp.replace('prcp_ann','Annual')
    anoms_all_prcp = anoms_all_prcp.replace('prcp_mths10-11','Oct,Nov')
    anoms_all_prcp = anoms_all_prcp.replace('prcp_mths12-1-2','Dec,Jan,Feb')
    anoms_all_prcp = anoms_all_prcp.replace('prcp_mths4-5','Apr,May')
    anoms_all_prcp = anoms_all_prcp.replace('prcp_mths6-7-8-9','Jun,Jul,Aug,Sep')
    
    anoms_all_tair = anoms_all_tair.set_index(['year','cluster','vname'])
    anoms_all_tair = anoms_all_tair.unstack('vname')
    anoms_all_prcp = anoms_all_prcp.set_index(['year','cluster','vname'])
    anoms_all_prcp = anoms_all_prcp.unstack('vname')
    
    i = 1
    for cname in anoms_all_tair.index.levels[1]:
                
        anoms_tair = anoms_all_tair.xs(cname,level='cluster').transpose()
        anoms_tair.index = anoms_tair.index.droplevel(0)
        anoms_tair.index.name = None
    
        anoms_prcp = anoms_all_prcp.xs(cname,level='cluster').transpose()
        anoms_prcp.index = anoms_prcp.index.droplevel(0)
        anoms_prcp.index.name = None
        


        ax = plt.subplot(12,2,i)
        i+=1
        sns.heatmap(anoms_tair,vmin=0,vmax=5,xticklabels=True, yticklabels=True,annot=True,fmt='.1f',annot_kws={'fontsize':7})
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.xlabel('')
        
        if cname != 'cluster12':
            plt.gca().get_xaxis().set_ticks([])
        
        
        ax = plt.subplot(12,2,i)
        i+=1
        
        sns.heatmap(anoms_prcp, cmap='RdBu', vmin=.6,vmax=1.4,xticklabels=True, yticklabels=True,annot=True,fmt='.2f',annot_kws={'fontsize':7})
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.xlabel('')
        plt.ylabel(cname)
        plt.gca().get_yaxis().set_ticks([])
        
        if cname != 'cluster12':
            plt.gca().get_xaxis().set_ticks([])
        
        
if __name__ == '__main__':
    
    clusters = pd.read_csv('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/cluster_nums.csv')
    clusters.columns = ['model_name','k.cluster']
    clusters = clusters.set_index('model_name')
    clusters['rcp'] = np.array([r.split('_')[1] for r in clusters.index])
    
    cnts = clusters.reset_index().groupby(['k.cluster','rcp']).count().reset_index()
    
    g = sns.factorplot(x="rcp", y="count", col='cluster', data=cnts,kind="bar",order=np.unique(clusters['rcp'].values),col_wrap=4)

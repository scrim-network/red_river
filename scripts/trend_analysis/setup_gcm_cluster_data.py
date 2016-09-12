'''
Script to setup data for cluster analysis of projected GCM precipitation and
temperature trends.
'''

import pandas as pd
import numpy as np
import esd
import os

if __name__ == '__main__':
    
    path_cmip5_trends = esd.cfg.path_cmip5_trends
    
    anom_fpaths = [(os.path.join(path_cmip5_trends, 'anoms_tas_2006_2099_ann.csv'),'tair_ann'),
                   (os.path.join(path_cmip5_trends, 'anoms_tas_2006_2099_mths10-11.csv'),'tair_mths10-11'),
                   (os.path.join(path_cmip5_trends, 'anoms_tas_2006_2099_mths12-1-2.csv'),'tair_mths12-1-2'),
                   (os.path.join(path_cmip5_trends, 'anoms_tas_2006_2099_mths4-5.csv'),'tair_mths4-5'),
                   (os.path.join(path_cmip5_trends, 'anoms_tas_2006_2099_mths6-7-8-9.csv'),'tair_mths6-7-8-9'),
                   (os.path.join(path_cmip5_trends, 'anoms_pr_2006_2099_ann.csv'),'prcp_ann'),
                   (os.path.join(path_cmip5_trends, 'anoms_pr_2006_2099_mths10-11.csv'),'prcp_mths10-11'),
                   (os.path.join(path_cmip5_trends, 'anoms_pr_2006_2099_mths12-1-2.csv'),'prcp_mths12-1-2'),
                   (os.path.join(path_cmip5_trends, 'anoms_pr_2006_2099_mths4-5.csv'),'prcp_mths4-5'),
                   (os.path.join(path_cmip5_trends, 'anoms_pr_2006_2099_mths6-7-8-9.csv'),'prcp_mths6-7-8-9'),]
    
    anoms_all = []
    
    for fpath,vname in anom_fpaths:
        
        print vname
        
        anoms = pd.read_csv(fpath, index_col='time')
        anoms.index = pd.DatetimeIndex(anoms.index)
        # resample to 10-year means for the cluster analysis
        anoms = anoms.resample('10AS').mean()
        anoms = anoms.stack()
        anoms.index.names = ['time', 'model_name']
        anoms = anoms.unstack('time')
        anoms.columns = [vname + "_" + yr for yr in anoms.columns.year.astype(np.str)]
        
        anoms_all.append(anoms)
        
    anoms_all = pd.concat(anoms_all,axis=1)
    
    # Remove model realizations that don't have both tair and prcp
    print ("Removing model realizations that don't have both tair and prcp: " +
           str(anoms_all.index[anoms_all.isnull().sum(axis=1) > 0]))
    anoms_all = anoms_all.dropna()
    anoms_all.index.name = 'model_name'
    anoms_all.to_csv(os.path.join(esd.cfg.path_cmip5_trends, 'cluster_anoms.csv'))
    
    
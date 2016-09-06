import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    anom_fpaths = [('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_tas_2006_2099_ann.csv','tair_ann'),
                   ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_tas_2006_2099_mths10-11.csv','tair_mths10-11'),
                   ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_tas_2006_2099_mths12-1-2.csv','tair_mths12-1-2'),
                   ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_tas_2006_2099_mths4-5.csv','tair_mths4-5'),
                   ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_tas_2006_2099_mths6-7-8-9.csv','tair_mths6-7-8-9'),
                   ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_pr_2006_2099_ann.csv','prcp_ann'),
                   ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_pr_2006_2099_mths10-11.csv','prcp_mths10-11'),
                   ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_pr_2006_2099_mths12-1-2.csv','prcp_mths12-1-2'),
                   ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_pr_2006_2099_mths4-5.csv','prcp_mths4-5'),
                   ('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/anoms_pr_2006_2099_mths6-7-8-9.csv','prcp_mths6-7-8-9'),]
    
    anoms_all = []
    
    for fpath,vname in anom_fpaths:
        
        print vname
        
        anoms = pd.read_csv(fpath, index_col='time')
        anoms.index = pd.DatetimeIndex(anoms.index)
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
    anoms_all.to_csv('/storage/home/jwo118/group_store/red_river/cmip5_trends_r2/cluster_anoms.csv')
    
    
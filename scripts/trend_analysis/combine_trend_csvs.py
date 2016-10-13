'''
Script to combine trend CSVs from calc_gcm_trends.py into a single CSV.
'''

import esd
import glob
import os
import pandas as pd

mth_num_to_letter = {1:"J",2:'F',3:'M',4:'A',5:'M',6:'J',
                     7:'J',8:'A',9:'S',10:'O',11:'N',12:'D'}

def fname_to_season(fname):
    
    season = fname.split('.')[0].split('_')[-1]
    
    if season == 'ann':
        season_name = 'ANNUAL'
    else:
        season_name = "".join([mth_num_to_letter[int(i)]
                               for i in season[4:].split('-')])
        
    return season_name
        
if __name__ == '__main__':
    
    # Trends for CMIP5 realizations have NOT been biased corrected and downscaled
    # path_trends = esd.cfg.path_cmip5_trends
    
    # Trends for CMIP5 realizations have been biased corrected and downscaled
    path_trends = os.path.join(esd.cfg.path_cmip5_trends, 'downscaled')
    
    fpaths = sorted(glob.glob(os.path.join(path_trends, 'pr_*.csv')))
    fpaths.extend(sorted(glob.glob(os.path.join(path_trends, 'tas_*.csv'))))
    fpath_out = os.path.join(path_trends, 'red_river_trends.csv')
    
    df_all = []
    
    for fpath in fpaths:
        
        print fpath
        
        fname = os.path.basename(fpath)
        
        df = pd.read_csv(fpath, header=None, names=['model_run', 'trend'], index_col=0)
        df = df.reset_index()
        df['rcp'] = df['model_run'].str.split('_',expand=True)[1].values
        df['season'] = fname_to_season(fname)
        df['elem'] = fname.split('_')[0]
        df['time_period'] = "_".join(fname.split('_')[1:3])
        df = df[['model_run','rcp','season','elem','time_period','trend']]
        
        df_all.append(df)
    
    df_all = pd.concat(df_all, ignore_index=True)
    
    # Get rid of duplicate trends from models that didn't have complete time
    # period
    df_all = df_all[~df_all.trend.duplicated(keep=False)]
    df_all.to_csv(fpath_out,index=False)

'''
Script to calculate aggregate annual and seasonal trends for CMIP5 GCM projections
in the Red River basin.
'''

import esd
import glob
import numpy as np
import os
import pandas as pd
import statsmodels.formula.api as smf
import xarray as xr

def fpaths_runnames(path_cmip5, fname_pattern, min_year, max_year):
    
    paths_rcps = sorted(glob.glob(os.path.join(path_cmip5, 'historical*_merged')))
     
    fpaths = []
     
    for path_rcp in paths_rcps:
         
        fpaths.extend(sorted(glob.glob(os.path.join(path_rcp, fname_pattern))))
        
    fpaths = np.array(fpaths)
     
    def parse_run_name(fpath):
         
        try:
            vals = os.path.basename(fpath).split('.')
             
            model_name = vals[2]
            rcp_name = vals[3].split('+')[1]
            realization_name = vals[4]
             
            return "_".join([model_name,rcp_name,realization_name])
        except Exception as e:
            return None
        
    def parse_start_end_yr(fpath):
        
        return [int(i[0:4])
                for i in os.path.basename(fpath).split('.')[-2].split('-')]
         
    run_names = pd.Series([parse_run_name(fpath) for fpath in fpaths])
     
    yrs = pd.DataFrame([parse_start_end_yr(fpath) for fpath in fpaths],
                       columns=['start_yr','end_yr'])
    
    mask_incmplt = (~((yrs.start_yr <= min_year) & (yrs.end_yr >= max_year)))
    mask_rm = ((run_names.isnull()) | (mask_incmplt)).values
    
    print "Removing the following fpaths: "+str(fpaths[mask_rm])
     
    fpaths = fpaths[~mask_rm]
    run_names = run_names[~mask_rm]
    
    return fpaths, run_names
    
def calc_trend(anoms, vname):
    
    anoms = pd.DataFrame(anoms)
    anoms['i'] = np.arange(len(anoms))
    
    lm = smf.ols(formula='%s ~ i'%vname, data=anoms).fit()
    
    return lm.params.i

def output_ann_trends(path_cmip5, fname_pattern, vname, unit_convert_func,
                      resample_func, trend_func, anom_func, start_base,
                      end_base, start_trend, end_trend, path_out):
    
    fpaths, run_names = fpaths_runnames(path_cmip5, fname_pattern,
                                        pd.Timestamp(start_base).year,
                                        pd.Timestamp(end_trend).year)
    
    trends_all = pd.Series(np.nan,index=run_names)
    anoms_all = []
    
    for run_name,fpath in zip(run_names, fpaths):
        
        #print run_name
        
        ds = xr.open_dataset(fpath)
        mask_lat = np.logical_and(ds.lat.values >= esd.cfg.bbox[1],
                                  ds.lat.values <= esd.cfg.bbox[-1])
        mask_lon = np.logical_and(ds.lon.values >= esd.cfg.bbox[0],
                                  ds.lon.values <= esd.cfg.bbox[2]) 
        da = ds[vname][:, mask_lat, mask_lon].load()
        ds.close()
            
        mthly = da.resample('1MS', dim='time', how=resample_func)
        ann = mthly.resample('AS', dim='time', how=resample_func)
        
        norms = ann.loc[start_base:end_base].mean('time')
        anoms = anom_func(ann, norms)
        anoms.name = vname
        
        anoms = anoms.mean(['lat', 'lon']).to_dataframe()
        anoms = anoms.loc[start_trend:end_trend]
        trend = trend_func(calc_trend(anoms, vname))  

        anoms_all.append(pd.DataFrame({run_name:anoms[vname]}))
        trends_all.loc[run_name] = trend
    
    anoms_all = pd.concat(anoms_all, axis=1)
    fname_out = "anoms_%s_%s_%s_ann.csv"%(vname,start_trend,end_trend)
    anoms_all.to_csv(os.path.join(path_out, fname_out))
    
    trends_all.index.name = 'model_run'
    fname_out = "%s_%s_%s_ann.csv"%(vname,start_trend,end_trend)
    trends_all.to_csv(os.path.join(path_out,fname_out))

def output_season_trends(path_cmip5, fname_pattern, season_mths, vname,
                         unit_convert_func, resample_func, anom_func, trend_func,
                         start_base, end_base, start_trend, end_trend, path_out,
                         rolling=False):
    
    fpaths, run_names = fpaths_runnames(path_cmip5, fname_pattern, pd.Timestamp(start_base).year, pd.Timestamp(end_trend).year)
    
    trends_all = pd.Series(np.nan,index=run_names)
    anoms_all = []
    
    for run_name,fpath in zip(run_names, fpaths):
        
        #print run_name
        
        ds = xr.open_dataset(fpath)
        mask_lat = np.logical_and(ds.lat.values >= esd.cfg.bbox[1],
                                  ds.lat.values <= esd.cfg.bbox[-1])
        mask_lon = np.logical_and(ds.lon.values >= esd.cfg.bbox[0],
                                  ds.lon.values <= esd.cfg.bbox[2]) 
        da = ds[vname][:, mask_lat, mask_lon].load()
        ds.close()
        
        mthly = da.resample('1MS', dim='time', how=resample_func)

        mask_season = np.in1d(mthly['time.month'].values, season_mths)
        
        if rolling:
        
            mthly = mthly.where(xr.DataArray(mask_season, coords=[mthly.time]))
            roller = mthly.rolling(min_periods=len(season_mths), center=True, time=len(season_mths))
            mthly = getattr(roller, resample_func)()
        
        else:
            
            mthly = mthly[mask_season]
        
        ann_season = mthly.resample('AS',dim='time', how=resample_func)
        
        norms = ann_season.loc[start_base:end_base].mean('time')
        anoms = anom_func(ann_season, norms)
        anoms.name = vname
        
        anoms = anoms.mean(['lat','lon']).to_dataframe()
        anoms = anoms.loc[start_trend:end_trend]
        trend = trend_func(calc_trend(anoms, vname))  
        
        anoms_all.append(pd.DataFrame({run_name:anoms[vname]}))
        trends_all.loc[run_name] = trend
    
    anoms_all = pd.concat(anoms_all, axis=1)
    fname_out = "anoms_%s_%s_%s_mths%s.csv"%(vname,start_trend,end_trend,
                                             "-".join([str(x) for x in season_mths]))
    anoms_all.to_csv(os.path.join(path_out, fname_out))
    
    
    trends_all.index.name = 'model_run'
    fname_out = "%s_%s_%s_mths%s.csv"%(vname,start_trend,end_trend,
                                       "-".join([str(x) for x in season_mths]))
    trends_all.to_csv(os.path.join(path_out,fname_out))

if __name__ == '__main__':
    
    # Trends for CMIP5 realizations have NOT been biased corrected and downscaled
    path_cmip5_orig = esd.cfg.path_cmip5_cleaned
    path_out_orig = os.path.join(esd.cfg.path_cmip5_trends, 'orig')
    
    # Trends for CMIP5 realizations have been biased corrected but NOT downscaled
    path_cmip5_bc = esd.cfg.path_cmip5_debiased
    path_out_bc = os.path.join(esd.cfg.path_cmip5_trends, 'bias_corrected')
    
    # Trends for CMIP5 realizations have been biased corrected and downscaled
    path_cmip5_ds = esd.cfg.path_cmip5_downscaled
    path_out_ds = os.path.join(esd.cfg.path_cmip5_trends, 'downscaled')
    
    paths_all = zip([path_cmip5_ds],
                    [path_out_ds])
    
    start_base = esd.cfg.start_date_baseline.strftime('%Y-%m-%d')
    end_base = esd.cfg.end_date_baseline.strftime('%Y-%m-%d')
    trend_periods = [('2006','2050'), ('2006','2099')]
    
    # Prcp ann
    for path_cmip5,path_out in paths_all:
        
        print "#######################"
        print "Processing CMIP5 realizations for: " + path_cmip5
        print "#######################"
        
        for a_start, a_end in trend_periods:
            
            print "Processing Prcp ann over %s to %s"%(a_start, a_end)
            
            output_ann_trends(path_cmip5,
                              fname_pattern='pr.day*.nc',
                              vname='pr',
                              unit_convert_func=None,
                              resample_func='sum',
                              trend_func=lambda x: x*100*10, #% per decade,
                              anom_func= lambda mthly_grp,norms: mthly_grp / norms,
                              start_base=start_base,
                              end_base=end_base,
                              start_trend=a_start,
                              end_trend=a_end,
                              path_out=path_out)
        
        
        # Prcp seasonal
        for a_start, a_end in trend_periods:
            
            for season,use_roll in [([6,7,8,9], False),
                                    ([12,1,2], True),
                                    ([10,11], False),
                                    ([4,5], False),]:
                
                print "Processing Prcp seasonal %s over %s to %s"%(str(season), a_start, a_end)
                
                output_season_trends(path_cmip5,
                                     fname_pattern='pr.day*.nc',
                                     season_mths=season,
                                     vname='pr',
                                     unit_convert_func=None,
                                     resample_func='sum',
                                     anom_func=lambda mthly_grp,norms: mthly_grp / norms,
                                     trend_func=lambda x: x*100*10, #% per decade
                                     start_base=start_base,
                                     end_base=end_base,
                                     start_trend=a_start,
                                     end_trend=a_end,
                                     path_out=path_out, rolling=use_roll)
                
        for a_start, a_end in trend_periods:
             
            print "Processing Tair ann over %s to %s"%(a_start, a_end)
             
            output_ann_trends(path_cmip5,
                              fname_pattern='tas.day*.nc',
                              vname='tas',
                              unit_convert_func=None,
                              resample_func='mean',
                              trend_func=lambda x: x*10, #degC per decade,
                              anom_func= lambda mthly_grp,norms: mthly_grp - norms,
                              start_base=start_base,
                              end_base=end_base,
                              start_trend=a_start,
                              end_trend=a_end,
                              path_out=path_out)
        
        
        # Tair seasonal
        for a_start, a_end in trend_periods:
            
            for season,use_roll in [([6,7,8,9], False),
                                    ([12,1,2], True),
                                    ([10,11], False),
                                    ([4,5], False),]:
                
                print "Processing Tair seasonal %s over %s to %s"%(str(season), a_start, a_end)
                
                output_season_trends(path_cmip5,
                                     fname_pattern='tas.day*.nc',
                                     season_mths=season,
                                     vname='tas',
                                     unit_convert_func=None,
                                     resample_func='mean',
                                     anom_func=lambda mthly_grp,norms: mthly_grp - norms,
                                     trend_func=lambda x: x*10, #degC per decade,
                                     start_base=start_base,
                                     end_base=end_base,
                                     start_trend=a_start,
                                     end_trend=a_end,
                                     path_out=path_out, rolling=use_roll)

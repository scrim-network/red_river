'''
Script to plot historical annual cycles of tair and prcp for 
APHRODITE and CMIP5 realizations.
'''

import esd
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
sns.set_context('poster',font_scale=1.3, rc={"lines.linewidth": 3})
sns.set_style('whitegrid')

if __name__ == '__main__':
    
    # Precipitation
    
    df_prcp = pd.read_csv(os.path.join(esd.cfg.path_cmip5_trends, 'hist_ann_cycle_prcp.csv'),index_col=0)
    df_prcp.iloc[:,0:-1].plot(legend=False, color='#bdbdbd')
    plt.plot(df_prcp.aphro, color='#d95f02', label='APHRODITE',lw=6)
    plt.plot(df_prcp.iloc[:,0:-1].mean(axis=1), color='#7570b3', label='ensemble mean',lw=6)
    
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    
    plt.legend(handles[-3:],['ensemble member']+labels[-2:])
    plt.ylabel('precipitation (mm)')

    plt.tight_layout()
    plt.savefig(os.path.join(esd.cfg.path_cmip5_trends,'figures','annual_cycle_prcp.png'),
                dpi=300, bbox_inches='tight')
    
    # Temperature
    
    df_tair = pd.read_csv(os.path.join(esd.cfg.path_cmip5_trends, 'hist_ann_cycle_tair.csv'),index_col=0)
    df_tair.iloc[:,0:-1].plot(legend=False, color='#bdbdbd')
    plt.plot(df_tair.aphro, color='#d95f02', label='APHRODITE',lw=6)
    plt.plot(df_tair.iloc[:,0:-1].mean(axis=1), color='#7570b3', label='ensemble mean',lw=6)
    
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    
    plt.legend(handles[-3:],['ensemble member']+labels[-2:])
    plt.ylabel(u'temperature (\u00b0C)')

    plt.tight_layout()
    plt.savefig(os.path.join(esd.cfg.path_cmip5_trends,'figures','annual_cycle_tair.png'),
                dpi=300, bbox_inches='tight')
    
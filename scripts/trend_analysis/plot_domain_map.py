from mpl_toolkits.basemap import Basemap
import esd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set_style('whitegrid')

if __name__ == '__main__':
    
    bbox = esd.cfg.bbox
        
    buf = 3
    m = Basemap(resolution='i',projection='tmerc', llcrnrlat=bbox[1]-buf,
                area_thresh=10000, urcrnrlat=bbox[-1]+buf,
                llcrnrlon=bbox[0]-buf, urcrnrlon=bbox[2]+buf,lon_0=105,lat_0=0)

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    llats = m.drawparallels(np.arange(0,30,5),labels=[1,0,0,0])
    llons = m.drawmeridians(np.arange(80,120,5),labels=[0,0,0,1])
    m.fillcontinents(color='#f0f0f0')
    
    s = m.readshapefile(os.path.join(esd.cfg.path_data_bbox,'redriver_1degbnds',
                                     'redriver_1degbnds'),'redriver_1degbnds',
                        color='#e41a1c',linewidth=1.25)
    
    s = m.readshapefile(os.path.join(esd.cfg.path_data_bbox, 'red_river'),
                        'red_river', color='#377eb8',linewidth=2)
    
    plt.savefig(os.path.join(esd.cfg.path_data_bbox,'figures',
                             'domain_map.png'), dpi=300, bbox_inches='tight')
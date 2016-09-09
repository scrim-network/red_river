from .misc import mkdir_p
from ConfigParser import ConfigParser
from copy import copy
import numpy as np
import os

class RedRiverConfig():
    
    def __init__(self, fpath_ini):
    
        cfg = ConfigParser()
        cfg.read(fpath_ini)
        
        self.data_root = cfg.get('REDRIVER_CONFIG', 'data_root')
        
        bbox_str = cfg.get('REDRIVER_CONFIG', 'domain_bounds_red_river')
        self.bbox = tuple([np.float(i) for i in bbox_str.split(',')])
        self.path_data_bbox = os.path.join(self.data_root, 'domain_bbox')
        mkdir_p(self.path_data_bbox)
        
        self.fpath_aphrodite_prcp = cfg.get('REDRIVER_CONFIG', 'FPATH_APHRODITE_PRCP')
        self.fpath_aphrodite_tair = cfg.get('REDRIVER_CONFIG', 'FPATH_APHRODITE_TAIR')

    def to_str_dict(self):
        
        a_dict = copy(self.__dict__)
        for a_key in a_dict.keys():
            a_dict[a_key] = str(a_dict[a_key])
        return a_dict
            
        
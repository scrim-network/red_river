import os

# rpy2
robjects = None
numpy2ri = None
r = None
ri = None
R_LOADED = False

def _load_R():

    global R_LOADED

    if not R_LOADED:
        
        global robjects
        global numpy2ri
        global r
        global ri
        
        # https://github.com/ContinuumIO/anaconda-issues/issues/152
        import readline
        
        import rpy2
        import rpy2.robjects
        robjects = rpy2.robjects
        r = robjects.r
        import rpy2.rinterface
        ri = rpy2.rinterface
        
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        
        path_root = os.path.dirname(__file__)
        fpath_rscript = os.path.join(path_root, 'rpy', 'debias.R')

        r.source(fpath_rscript)
        
        R_LOADED = True
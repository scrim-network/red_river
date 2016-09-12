# Script to load configuration parameters from python
###############################################################################

library(rPython)
python.exec("import esd")
cfg <- python.get('esd.cfg.to_str_dict()')

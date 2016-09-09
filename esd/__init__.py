from . import util

from .util.config import RedRiverConfig
import os
cfg = RedRiverConfig(os.getenv('REDRIVER_INI'))
#########
Red River CMIP5 bias correction and downscaling
#########

Python and R code for bias correction and empirical-statistical downscaling of
CMIP5 projections for the Red River basin in Vietnam. Bias correction and downscaling of
CMIP5 projections was conducted in support of:

Quinn JD, Reed PM, Giuliani M, Castelletti A, Oyler JW, Nicholas RE. 2018. Exploring How
Changing Monsoonal Dynamics and Human Pressures Challenge Multi-Reservoir Management for
Flood Protection, Hydropower Production and Agricultural Water Supply.
Water resources research. DOI: `10.1029/2018WR022743. <http://dx.doi.org/10.1029/2018WR022743>`_

=======================
Bias Correction Methods
=======================

We use a variation of the modified equidistant quantile matching (EDCDFm)
algorithm (`Li et al. 2010`_) of (`Pierce et al. 2015`_) to bias correct CMIP5
projections for the Red River, Vietnam. Details of the Red River variation of
EDCDFm are provided `here. <https://github.com/scrim-network/red_river/blob/master/docs/bias_correction_methods.ipynb>`_

=======================
Downscaling Methods
=======================

We use a variation of the constructed analogs method described by (`Pierce et al. 2014`_)
to downscale CMIP5 projections for the Red River, Vietnam. Our modified method is
termed **Constructed Analogs with Single Anomaly Analog (CASAA)**. Details of the
Red River CASAA method are provided `here. <https://github.com/scrim-network/red_river/blob/master/docs/downscaling_methods.ipynb>`_

.. _Pierce et al. 2015: http://dx.doi.org/10.1175/JHM-D-14-0236.1
.. _Li et al. 2010: http://dx.doi.org/10.1029/2009JD012882
.. _Pierce et al. 2014: http://dx.doi.org/10.1175/JHM-D-14-0082.1

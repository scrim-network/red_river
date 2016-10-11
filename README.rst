#########
Red River
#########

Python and R code for bias correction and empirical-statistical downscaling of
GCM projections for the Red River basin in Vietnam. 

=======================
Bias Correction Methods
=======================

We use a variation of the modified equidistant quantile matching (EDCDFm)
algorithm (`Li et al. 2010`_) of (`Pierce et al. 2015`_) to bias correct CMIP5
projections for the Red River, Vietnam. Key differences from (`Pierce et al. 2015`_)
include:

* We only apply EDCDFm over wet days when bias correcting precipitation. Unlike
  (`Pierce et al. 2015`_), this allows a model-predicted increase in number of wet
  days to be preserved and keeps models with too many historical dry days from
  being changed to all wet days over the historical period when biased corrected.
* We do not apply the frequency-dependent bias correction method described in
  section 4 of (`Pierce et al. 2015`_).
* If bias correction is applied without a moving time window, it will correct
  extremes, but not the annual cycle. Conversely, if bias correction is applied
  over a small moving window, it will correct the annual cycle, but may not
  correctly represent extremes. This is especially true for the EDCDFm method.
  To balance the correction of extremes and the annual cycle, (`Pierce et al. 2015`_)
  use a “preconditioning” step and an iterative application of EDCDFm over
  different moving window sizes (section 5). We use an alternative method. We
  first bias correct with EDCDFm using a 31-day moving window. We then bias correct
  with EDCDFm without a moving window. Finally, we bias correct the values from
  the 31-day moving window bias correction using those from the bias correction
  without a moving window.

Details of the Red River variation of EDCDFm in a `project pdf`_.

.. _Pierce et al. 2015: http://dx.doi.org/10.1175/JHM-D-14-0236.1
.. _Li et al. 2010: http://dx.doi.org/10.1029/2009JD012882
"""
Module that contains P(lambda_true|M), to be used by
selection function models that use
P(lambda_obs|M) = P(lambda_obs|lambda_true)P(lambda_true|M)
"""

from .halo_mass_observable import HaloMassObservable
from .lognormal_powerlaw import LognormalPowerLawHaloMassObservable

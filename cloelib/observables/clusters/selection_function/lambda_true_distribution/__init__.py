"""
Module that contains P(lambda_true|M), to be used by
selection function models that use
P(lambda_obs|M) = P(lambda_obs|lambda_true)P(lambda_true|M)
"""

from .lambda_true_distribution import LambdaTrueDistribution
from .lognormal_powerlaw import LognormalPowerLawLambdaTrueDistribution

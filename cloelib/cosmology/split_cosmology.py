# cloelib imports
from cloelib.auxiliary import extrapolator
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.cosmology.derived_cosmology import growth_function_ODE

import numpy as np
from scipy import interpolate
import copy
from typing import Optional, Sequence

class SplitLinearPerturbations:
    """Class to output the rescaled linear matter power spectrum for the
    growth-geometry split"""

    def __init__(
        self, background: Background, omega_m_growth: float, redshifts: np.ndarray, perturbations: Perturbations,
    ):
        """Initialise SplitLinearPerturbations."""
        self.background = background
        self.omega_m_growth = omega_m_growth
        self.z = redshifts
        self.kmax = 100
        self.perturbations = perturbations

    @property
    def _interface_args(self) -> dict:
        """Save internal structure format of interface codes."""
        return self.interface_args

    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the growth-geometry split CLASS linear matter power
        spectrum. This implies a rescaling with the growth function of the
        matter power spectrum and also sigma_8, as in 2301.03694

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        hubble_units: (Optional) bool
            Flag to specify if output in h units, defaults to False

        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False

        Returns
        -------
        pk_linear_EBS: numpy.ndarray
            Linear matter power spectrum at the specified scale
            and redshift from the Einstein-Boltzmann solver. This is needed to
            compute the boost factor in the class SplitNonLinearPerturbations.

        pk_linear: numpy.ndarray
            Rescaled linear matter power spectrum at the specified scale
            and redshift.
        """
        if hubble_units or k_hunit:
            raise ValueError("This CLASS method does not yet support h-units")
        self.pk_linear_EBS = self.perturbations.matter_power_spectrum(zs,ks)  # type:ignore[union-attr]

        omega_m_geo = self.background.Omega_cdm0 + self.background.Omega_b0

        # Compute the growth factor
        g_z_geo = growth_function_ODE(self.background, zs, omega_m_geo)
        g_z_growth = growth_function_ODE(self.background, zs, self.omega_m_growth)

        self.pk_linear = np.zeros_like(self.pk_linear_EBS)

        # Rescale the matter power spectrum with G(z)
        for i in range(len(zs)):
            rescale_fac = g_z_growth[i] ** 2 / g_z_geo[i] ** 2
            self.pk_linear[i, :] = rescale_fac * self.pk_linear_EBS[i, :]

        # Rescale sigma_8
        self.sigma8_0 = g_z_growth[i] / g_z_geo[i] * self.sigma8_0_EBS()
        return self.pk_linear

    def sigma8_0_EBS(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology from the EBS.

        Returns:
        --------
        float
            The sigma8 value.
        """
        return self.perturbations.sigma8_0()


class SplitNonLinearPerturbations:
    """Class to output the rescaled non-linear matter power spectrum for the
    growth-geometry split"""

    def __init__(
        self,
        redshifts: np.ndarray,
        pk_linear: np.ndarray,
        perturbations_lin: Perturbations,
        perturbations_NL: Perturbations,
    ):
        """Initialize the OmgrowthLinearPerturbation and OmgrowthNonLinearPerturbation instance."""        
        self.z = redshifts
        self.kmax = 100
        self.pk_linear = pk_linear
        self.perturbations_lin = perturbations_lin
        self.perturbations_NL = perturbations_NL

    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        """Calculate the split CLASS non-linear matter power spectrum.

        Parameters
        ----------
        zs: numpy.ndarray
            redshifts

        ks: numpy.ndarray
            wavenumber

        hubble_units: (Optional) bool
            Flag to specify if output in h units, defaults to False

        k_hunit: (Optional) bool
            Flag to specify if wavenumber in h units, defaults to False

        Returns
        -------
        pk: numpy.ndarray
            Non-linear matter power spectrum at the specified scale
            and redshift
        """

        if hubble_units or k_hunit:
            raise ValueError("This CLASS method does not yet support h-units")
        pk_linear_growth = self.perturbations_lin.matter_power_spectrum(zs,ks)  # type:ignore[union-attr]
        pk_nonlinear_growth = self.perturbations_NL.matter_power_spectrum(zs,ks)  # type:ignore[union-attr]

        # Compute the boost factor
        boost = pk_nonlinear_growth / pk_linear_growth

        # Add the boost to the rescaled power spectrum
        return boost * self.pk_linear

"""Defining a general perturbations class with a boost to be used for the Weyl potential measurement"""

# cloelib imports
from cloelib.cosmology.cosmology import Background, Perturbations

# General imports
import numpy as np
from typing import Union, TypeVar
import jax.numpy as jnp

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


class Weyl_Perturbations:
    """
    Simplified wrapper that implements the Perturbations-like interface by delegating to
    an inner `perturbations` object and applying a boost.

    Assumption: zs and ks passed into methods are array-like of the same array-library
    (both numpy arrays or both jax arrays), and their shapes are compatible for elementwise
    ops by the wrapped perturbations implementation.
    """

    def __init__(self, perturbations: Perturbations, redshifts: T, z_ini: float):
        self.perturbations = perturbations
        self.z = redshifts
        self.z_ini = float(z_ini)
        self.k = perturbations.k

    @property
    def background(self) -> Background:
        return self.perturbations.background

    def growth_factor(self, zs: T, ks: T) -> T:
        return self.perturbations.growth_factor(zs, ks)

    def growth_rate(self) -> T:
        # We multiply the growth rate by th growth factor (normalized at z_ini).
        # This is to include the RSD effect in the Weyl measurement (without modifying angular_two_point.py), while properly accounting for the growth at z_ini.
        zini_arr = self.z * 0 + self.z_ini
        gf = (
            self.perturbations.growth_factor(self.z, self.k[:1])[:, 0]
            / self.perturbations.growth_factor(zini_arr, self.k[:1])[:, 0]
        )
        return gf * self.perturbations.growth_rate()

    def boost(self, zs: T, ks: T) -> T:
        """
        boost(zs, ks) = growth_factor(zs, ks) / growth_factor(zs, k0_array)
        where k0_array has the same shape and dtype as ks, filled with k0.
        """
        # Create k0 array with same shape as ks; zini_arr with same shape as zs:
        k0 = self.k[0]
        k0_arr = ks * 0 + k0
        zini_arr = zs * 0 + self.z_ini

        gf_num = self.perturbations.growth_factor(
            zs, ks
        ) / self.perturbations.growth_factor(zini_arr, ks)
        gf_den = self.perturbations.growth_factor(
            zs, k0_arr
        ) / self.perturbations.growth_factor(zini_arr, k0_arr)

        return gf_num / gf_den

    def matter_power_spectrum(
        self, zs: T, ks: T, hubble_units: bool = False, k_hunit: bool = False
    ) -> T:
        """
        Return boosted matter power spectrum:
          P_boosted(zs, ks) = boost(zs, ks) * P_base(z_ini_array, ks)

        Here z_ini_array is an array matching zs (type & shape) where every entry == self.z_ini.
        """
        # create z_ini array matching zs' type & shape
        z_ini_arr = zs * 0 + self.z_ini

        # Evaluate the base power at z_ini_arr and ks
        P_base = self.perturbations.matter_power_spectrum(
            z_ini_arr, ks, hubble_units=hubble_units, k_hunit=k_hunit
        )

        b = self.boost(zs, ks)
        return b**2 * P_base

    def sigma8_0(self) -> float:
        return self.perturbations.sigma8_0()

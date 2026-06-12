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

    def __init__(
        self,
        perturbations_NL: Perturbations,
        perturbations_lin: Perturbations,
        redshifts: T,
        z_ini: float,
    ):
        self.perturbations_lin = perturbations_lin
        self.perturbations_NL = perturbations_NL
        self.z = redshifts  # Note: These are the redshifts at which the Weyl potential measurement will be performed.
        self.z_ini = float(z_ini)

        # Check that z_ini is smaller or equal to the maximum redshift in perturbations_NL.z, otherwise print error
        z_max = np.max(self.perturbations_NL.z)
        if z_max < self.z_ini:
            raise ValueError(
                f"z_ini={self.z_ini} is larger than the maximum available redshift "
                f"in perturbations_NL.z (z_max={z_max})."
            )

        # Check if perturbations_NL.k and perturbations_lin.k are the same, otherwise print warning (could lead to errors when calculating the boost factor)
        if not np.array_equal(perturbations_NL.k, perturbations_lin.k):
            print(
                "Warning: perturbations_NL.k and perturbations_lin.k are not the same."
            )

        self.k = perturbations_NL.k

    @property
    def background(self) -> Background:
        bg_lin = self.perturbations_lin.background
        bg_nl = self.perturbations_NL.background

        if bg_lin is not bg_nl:
            raise ValueError("Background objects must be the same instance.")

        return bg_lin

    def growth_factor(self, zs: T, ks: T) -> T:
        return self.perturbations_NL.growth_factor(zs, ks)

    def growth_rate(self) -> T:
        # Note: Current implementations of CAMB/CLASS pertrubations classes do not allow to specify a zs argument for growth_rate(), it is always calculated at self.z;

        z_target = np.asarray(
            self.z
        )  # These are the redshifts at which the Weyl potential measurement will be performed, and at which the growth rate will be calculated for the RSD contribution to Cell.
        z_source = np.asarray(
            self.perturbations_NL.z
        )  # These are the redshifts at which the growth rate is calculated in the perturbations_NL object.

        # Build pairwise comparison matrix
        matches = np.isclose(z_target[:, None], z_source[None, :], rtol=0.0, atol=1e-12)

        # Check that every target z has at least one match
        if not np.all(matches.any(axis=1)):
            missing = z_target[~matches.any(axis=1)]
            raise ValueError(
                f"Some redshifts in self.z are not in perturbations_NL.z: {missing}"
            )

        # Take first match along each row → indices in z_source which match z_target entries
        indices = np.argmax(matches, axis=1)

        gr = (
            self.perturbations_NL.growth_rate()
        )  # This is the growth rate at all redshifts in perturbations_NL.z.
        return gr[
            indices
        ]  # This is returning the growth rate at the redshifts corresponding to self.z.

    def matter_power_spectrum(self, zs: T, ks: T) -> T:
        """
        Return boosted matter power spectrum:
          P_boosted(zs, ks) = boost(zs, ks) * P_base(z_ini_array, ks)

        Here z_ini_array is an array matching zs (type & shape) where every entry == self.z_ini.
        """

        # create z_ini array matching zs' type & shape
        z_ini_arr = zs * 0 + self.z_ini

        # Evaluate the base power spectrum at z_ini_arr and ks
        P_base = self.perturbations_lin.matter_power_spectrum(z_ini_arr, ks)

        # Evaluate the linear and nonlinear power spectra at zs and ks
        pk_linear_growth = self.perturbations_lin.matter_power_spectrum(zs, ks)
        pk_nonlinear_growth = self.perturbations_NL.matter_power_spectrum(zs, ks)

        boost = pk_nonlinear_growth / pk_linear_growth

        # Multiply the base power spectrum by the boost factor
        return boost * P_base

    def matter_power_spectrum_cb(self, zs: T, ks: T) -> T:
        """
        Added for consistency with perturbations protocol; we apply a boost equivalently to the implementation in matter_power_spectrum.

        Return boosted matter power spectrum of cold dark matter + baryons:
          P_boosted(zs, ks) = boost(zs, ks) * P_base(z_ini_array, ks)

        Here z_ini_array is an array matching zs (type & shape) where every entry == self.z_ini.
        """

        # create z_ini array matching zs' type & shape
        z_ini_arr = zs * 0 + self.z_ini

        # Evaluate the base power spectrum at z_ini_arr and ks
        P_base = self.perturbations_lin.matter_power_spectrum_cb(z_ini_arr, ks)

        # Evaluate the linear and nonlinear power spectra at zs and ks
        pk_linear_growth = self.perturbations_lin.matter_power_spectrum_cb(zs, ks)
        pk_nonlinear_growth = self.perturbations_NL.matter_power_spectrum_cb(zs, ks)

        boost = pk_nonlinear_growth / pk_linear_growth
        P_base = self.perturbations_NL.matter_power_spectrum(z_ini_arr, ks)

        # Evaluate the linear and nonlinear power spectra at zs and ks
        pk_linear_growth = self.perturbations_lin.matter_power_spectrum(zs, ks)
        pk_nonlinear_growth = self.perturbations_NL.matter_power_spectrum(zs, ks)

        # Compute the boost factor
        boost = pk_nonlinear_growth / pk_linear_growth

        # Multiply the base power spectrum by the boost factor
        return boost * P_base

    def sigma8_0(self) -> float:
        return self.perturbations_NL.sigma8_0()

    def sigma8_zini(self) -> float:
        # Calculate sigma8 at z_ini using the linear growth factor
        k = np.array(
            [0.1]
        )  # roughly corresponds to the scales probed by sigma8; exact k value does not matter as we're using the linear perturbation function
        D_zini = self.perturbations_lin.growth_factor(np.array([self.z_ini]), k)[
            0, 0
        ]  # this is already normalized to 1 at z=0
        sigma8_zini = self.sigma8_0() * D_zini
        return sigma8_zini

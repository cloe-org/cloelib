# General imports
from typing import Protocol, TypeVar, Union, runtime_checkable

import jax.numpy as jnp
import numpy as np  # type: ignore

"""
Protocol for the halo mass density profiles.
"""

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


@runtime_checkable
class HaloProfile(Protocol):
    def _f_term(self, x: float) -> float:
        r"""Computes the F term."""
        ...

    def _g_term(self, x: float) -> float:
        r"""Computes the G term."""
        ...

    def _surface_mass_density_1h(self, R: T, RDelta: T, Delta: T, c: float):
        r"""Surface mass density profile.

        Computes the surface mass density profile at radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma: np.ndarray
            Surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        ...

    def _mean_surface_mass_density_1h(self, R: T, RDelta: T, Delta: T, c: float):
        r"""
        Mean surface mass density profile.

        Computes the mean surface mass density
        within a radius R.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        RDelta: np.ndarray
            Overdensity radius (units : Mpc / h).
        Delta: np.ndarray
            Critical overdensity.
        c: float
            Concentration.

        Returns
        -------
        Sigma_mean: np.ndarray
            Mean surface mass density (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        ...

    def surface_mass_density(
        self,
        R: T,
        z: T,
        M: T,
        c: float,
        halo_bias: str,
        radius_units: str,
    ):
        r"""
        Total surface mass density profile.

        Computes the total surface mass density profile at radius R,
        including the contribution from 2-halo term.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        M: np.ndarray
            Mass (Msun / h).
        c: float
            Concentration.
        halo_bias: np.ndarray (optional)
            Halo bias used for the 2h term, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        Sigma: np.ndarray
            Surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        ...

    def excess_surface_mass_density(
        self,
        R: T,
        z: T,
        M: T,
        c: float,
        halo_bias: str,
        radius_units: str,
    ):
        r"""
        Total excess surface mass density profile.

        Computes the total excess surface mass density profile at radius R,
        including the contribution from 2-halo term.

        Parameters
        ----------
        R: np.ndarray
            Radial points (units : Mpc / h)
        z: np.ndarray
            Redshift.
        c: float
            Concentration.
        M: np.ndarray
            Mass (Msun / h).
        halo_bias: np.ndarray (optional)
            Halo bias used for the 2h term, with shape (z.size, M.size).
        radius_units: str
            Unit for the input radius. Accepted values are:
            "Mpc/h", "radians", "degrees", "arcmin", "arcsec".

        Returns
        -------
        DeltaSigma: np.ndarray
            Excess surface mass density profile (units : h * Msun / pc**2).
            Shape: (z.size, M.size, R.size).
        """
        ...

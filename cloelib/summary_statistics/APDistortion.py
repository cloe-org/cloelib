"""Module to compute Alcock-Paczynski (AP) distortions parameters."""

# cloelib imports
from cloelib.cosmology.cosmology import Background

# General imports
from typing import Union, TypeVar
import numpy as np  # type: ignore
import jax.numpy as jnp

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


class APDistortion:
    """Class to compute AP distortion parameters from cosmological background quantities."""

    def __init__(self, background: Background, background_fiducial: Background):
        """
        Initialize the class instance.

        Parameters
        ----------
        background: Background
            Background class for computing background distances
        background_fiducial: Background
            Background class for computing fiducial background distances
        """
        self.background = background
        self.background_fiducial = background_fiducial

    def q_AP_tr(self, z: T) -> T:
        r"""AP distortion parameter transversal to the line of sight.

        .. math::
            q_{\perp}(z) &= \frac{D_{\rm M}(z)}{D_{\rm M,fid}(z)}\\
        Parameters
        ----------
        z: np.ndarray
           Redshift
        Returns
        -------
        q_tr: np.ndarray
           Transversal AP parameter
        """
        return self.background.angular_diameter_distance(
            z
        ) / self.background_fiducial.angular_diameter_distance(z)

    def q_AP_lo(self, z: T) -> T:
        r"""AP distortion parameter parallel to the line of sight.

        .. math::
            q_{\parallel}(z) &= \frac{H_{\rm fid}(z)}{H(z)}\\
        Parameters
        ----------
        z: np.ndarray
           Redshift
        Returns
        -------
        q_tr: np.ndarray
           Parallel AP parameter
        """
        return self.background_fiducial.hubble_parameter(
            z
        ) / self.background.hubble_parameter(z)

    def gamma_tr(self, z_true: T, z_meas: T) -> T:
        r"""Distortion parameter due to line misidentification perpendicular
        to the line of sight

        ..math::
            \gamma_\perp(z_{\rm true},z_{\rm meas}) =
            \frac{D_{\rm M,fid}(z_{\rm meas})}{D_{\rm M,fid}(z_{\rm true})}\\
        Parameters
        ----------
        z_true: np.ndarray
            True redshift
        z_meas: np.ndarray
            Measured redshift
        Returns
        -------
        gamma_tr: np.ndarray
            Perpendicular distortion parameter due to line misidentification
        """
        return (self.background_fiducial.angular_diameter_distance(z_true)
                / self.background_fiducial.angular_diameter_distance(z_meas)
                * (1.0 + z_true) / (1.0 + z_meas))

    def gamma_lo(self, z_true: T, z_meas: T) -> T:
        r"""Distortion parameter due to line misidentification parallel
        to the line of sight

        ..math::
            \gamma_\parallel(z_{\rm true},z_{\rm meas}) =
            \frac{H_{\rm fid}(z_{\rm true})}{D_{\rm fid}(z_{\rm meas})}\\
        Parameters
        ----------
        z_true: np.ndarray
            True redshift
        z_meas: np.ndarray
            Measured redshift
        Returns
        -------
        gamma_tr: np.ndarray
            Parallel distortion parameter due to line misidentification
        """
        return (self.background_fiducial.hubble_parameter(z_meas)
                / self.background_fiducial.hubble_parameter(z_true)
                * (1.0 + z_true) / (1.0 + z_meas))

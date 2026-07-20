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

    def __init__(self, background: Background, background_fiducial: Background, h_units: bool = False):
        """
        Initialize the class instance.

        Parameters:
            background (Background): Background class for computing background distances
            background_fiducial (Background): Background class for computing fiducial background distances
        """
        self.background = background
        self.background_fiducial = background_fiducial
        self.h_units = h_units

    def q_AP_tr(self, z: T) -> T:
        r"""AP distortion parameter transversal to the line of sight.

        $$
            q_{\perp}(z) = \frac{D_{\rm M}(z)}{D_{\rm M,fid}(z)}\\
        $$

        Parameters:
            z (np.ndarray): Redshift
        Returns:
            q_tr (np.ndarray): Transversal AP parameter
        """
        if self.h_units:
            return self.background.angular_diameter_distance(
                z
                )*self.background.h / self.background_fiducial.angular_diameter_distance(
                    z
                    ) / self.background_fiducial.h
        else:
            return self.background.angular_diameter_distance(
                z
            ) / self.background_fiducial.angular_diameter_distance(z)

    def q_AP_lo(self, z: T) -> T:
        r"""AP distortion parameter parallel to the line of sight.

        $$
            q_{\parallel}(z) = \frac{H_{\rm fid}(z)}{H(z)}\\
        $$

        Parameters:
            z (np.ndarray): Redshift
        Returns:
            q_tr (np.ndarray): Parallel AP parameter
        """
        if self.h_units:
            return self.background_fiducial.hubble_parameter(
                z
            ) / self.background.hubble_parameter(z) * self.background.h / self.background_fiducial.h
        else:
            return self.background_fiducial.hubble_parameter(
                z
            ) / self.background.hubble_parameter(z)

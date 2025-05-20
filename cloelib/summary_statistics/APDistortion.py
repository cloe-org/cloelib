# cloelib imports
from cloelib.cosmology.cosmology import Background

# General imports
from typing import Union, TypeVar
import numpy as np  # type: ignore
import jax.numpy as jnp

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])

class APDistortion:
    r"""Class to compute Alcock–Paczynski (AP) distortion parameters and 
    BAO dilation parameters from cosmological background quantities.

    This class centralizes the computation of:
    - AP distortion factors (transverse and parallel)

    Parameters
    ----------
    background: Background
        Background class for computing background distances
    background_fiducial: Background
        Background class for computing fiducial background distances
    """
    def __init__(self, background: Background, background_fiducial: Background):
        self.background = background
        self.background_fiducial = background_fiducial

    def q_AP_tr(self, z: T) -> T:
        r"""AP distortion parameter transversal to the line of sight
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
        return (self.background.angular_diameter_distance(z)
                / self.background_fiducial.angular_diameter_distance(z))

    def q_AP_lo(self, z: T) -> T:
        r"""AP distortion parameter parallel to the line of sight
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
        return (self.background_fiducial.hubble_parameter(z)
                / self.background.hubble_parameter(z))

# General imports
from typing import Protocol, TypeVar, Union, runtime_checkable

import jax.numpy as jnp
import numpy as np  # type: ignore

"""
- Introducing a protocol for the halo statistics part that we might have many versions of it.
"""

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


@runtime_checkable
class SelectionFunction(Protocol):
    def window_z_richness_observed(
        self,
        lambda_obs_edges,
        z_obs_edges,
        lambda_true,
        z_true,
    ):
        r"""
        Computes the window function for observed redshift and richness bins, i. e.:


        ..math:
            W_{\Delta\lambda_{\rm obs}, \Delta z_{\rm obs}}(\lambda_{\rm true}, z_{\rm true}) =
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            \int_{\Delta z_{\rm obs}}d z_{\rm obs}
            P(\lambda_{\rm obs}, z_{\rm obs}|\lambda_{\rm true}, z_{\rm true})
            \frac{c(\lambda_{\rm true}, z_{\rm true})}{p(\rm obs}, z_{\rm obs})}



        Computes the integral over Delta_Lobs_NC and Delta_zobs_NC of
        1/Omega_tot * sum_alpha Omega_alpha*Pα(λobs|λtr,ztr)*Pα(zobs|λtr,ztr)/Pα(λobs,zobs)*Cα(λtr,ztr).
        Builds the interpolators over (ltr,ztr) for all bins in Lobs_NC and zobs_NC.

        Parameters
        ----------
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_true : numpy.ndarray
            True richness to compute the window.
        z_true : numpy.ndarray
            True redshift to compute the window.

        Returns
        -------
        numpy.ndarray
            Window function for observed redshift and richness bins.
            Dimensions: (lambda_obs_edges, z_obs_edges, lambda_true, z_true)
        """
        ...

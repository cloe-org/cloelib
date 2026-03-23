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
    def scatter_z_obs(self, lambda_obs, z):
        r"""
        Statistical uncertainty on the observed redshift.

        Computes the scatter of the observed redshift PDF
        at the requested true redshift and observed richness points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        lambda_obs: numpy.ndarray
            Observed richness points.

        Returns
        -------
        scatter_z_obs: numpy.ndarray
            scatter_z_obs[i,j] where i is the true redshift axis
            and j the observed richness axis
        """
        ...

    def window_z_richness_observed(
        self, z_obs_edges, lambda_obs_edges, z_true, mass, lambda_true
    ):
        r"""
        Computes the window function for observed redshift and richness bins, i. e.:


        ..math:
            W_{\Delta\lambda_{\rm obs}, \Delta z_{\rm obs}}(M, z_{\rm true}) =
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            \int_{\Delta z_{\rm obs}}d z_{\rm obs}
            P(\lambda_{\rm obs}, z_{\rm obs}|M, z_{\rm true})
            \frac{c(M, z_{\rm true})}{p(\rm obs}, z_{\rm obs})}

        which can be written in some cases as:

        ..math:
            W_{\Delta\lambda_{\rm obs}, \Delta z_{\rm obs}}(M, z_{\rm true}) =
            \int_{0}^{\infty}d\lambda_{\rm true}
            P(\lambda_{\rm true}|M, z_{\rm true})
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            \int_{\Delta z_{\rm obs}}d z_{\rm obs}
            P(\lambda_{\rm obs}, z_{\rm obs}|\lambda_{\rm true}, z_{\rm true})
            \frac{c(\lambda_{\rm true}, z_{\rm true})}{p(\rm obs}, z_{\rm obs})}

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_true : numpy.ndarray
            True redshift to compute the window.
        mass : numpy.ndarray
            Mass to compute the window.
        lambda_true : numpy.ndarray
            Values to be used for marginalization over true richness.

        Returns
        -------
        numpy.ndarray
            Window function for observed redshift and richness bins.
            Dimensions: (z_obs_edges, lambda_obs_edges, z_true, mass)
        """
        ...

    def window_z_observed(self, z_obs_edges, lambda_obs_edges, z_true):
        r"""Compute the window function of each observed redshift bin, given by:

        ..math:
            W_{\Delta z_{\rm obs}}(\lambda_{\rm obs}, z_{\rm true}) = \int_{\Delta z_{\rm obs}}dz_{\rm obs} P(z_{\rm obs}|\lambda_{\rm obs}, z_{\rm true})

        Parameters
        ----------
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_true : numpy.ndarray
            True redshift to compute the window.

        Returns
        -------
        window_z_obs : numpy.ndarray
            Integral of P(z_obs|lambda_obs, z_true) in z_obs bins.
            Dimensions: (z_obs_edges, lambda_obs_edges, z_true).
        """
        ...

    def window_richness_observed(
        self,
        lambda_obs_edges,
        z_true,
        mass,
        lambda_true,
    ):
        r"""Compute the window function of each observed richness bin, given by:

        ..math:
            W_{\Delta\lambda_{\rm obs}}(M, z_{\rm true}) =
            \int_{0}^{\infty}d\lambda_{\rm true}
            P(\lambda_{\rm true}|M, z_{\rm true})
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            P(\lambda_{\rm obs}|\lambda_{\rm true}, z_{\rm true})

        Parameters
        ----------
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_true : numpy.ndarray
            True redshift to compute the window.
        mass : numpy.ndarray
            Mass to compute the window.
        lambda_true : numpy.ndarray
            Values to be used for marginalization over true richness.

        Returns
        -------
        window_lambda_obs : numpy.ndarray
            Integral of P(lambda_obs|\lambda_{\rm true}, z_true) in lambda_obs bins.
            Dimensions: (lambda_obs_edges, z_true, \lambda_{\rm true}).
        """
        ...

"""Angular Correlation Function Protocol."""

from typing import Protocol, Tuple, Union
import jax.numpy as jnp


class AngularCorrelationFunction(Protocol):
    """
    Protocol for angular correlation function classes.

    This protocol defines a standard interface for computing
    real-space two-point correlation functions xi(theta). Implementing classes are expected to
    encapsulate any internal information needed to evaluate
    the correlation function, such as multipole grids (ells),
    wavenumber grids (ks), and angular power spectra (Cl).

    Notes
    -----
    Implementations may return a single correlation function xi(theta)
    (e.g., galaxy clustering or galaxy-galaxy lensing) or a tuple
    (xi_plus, xi_minus) for spin-2 tracers (e.g., cosmic shear).
    """

    def get_xi(
        self, theta: jnp.ndarray
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Compute the angular correlation function(s) at angle theta.

        Parameters
        ----------
        theta : jnp.ndarray
            Angular separation(s) in radians.

        Returns
        -------
        jnp.ndarray or (jnp.ndarray, jnp.ndarray)
            Correlation function(s) xi(theta) or (xi_+, xi_-).
        """
        ...

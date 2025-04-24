from typing import Protocol, Tuple, Union
import jax.numpy as jnp

class AngularCorrelationFunction(Protocol):

    def get_xi(
        self,
        theta: jnp.ndarray
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
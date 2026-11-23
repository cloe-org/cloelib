"""
Module implementing the thermal Sunyaev-Zel'dovich (tSZ) Compton-y tracer.

This class is compatible with the Tracer protocol. The Compton-y power spectra
come from a halo model (`cloelib.cosmology.godmax_cosmology`).
"""

# cloelib imports
from cloelib.auxiliary.math_utils import check_nonzero_redshifts
from cloelib.cosmology.cosmology import Perturbations
from cloelib.observables.photo.contributions import COMPTON_Y, check_halo_model_field

# General imports
import jax.numpy as np


class ComptonYContribution:
    """Compton-y kernel, the single term of `tSZTracer.get_window()`.

    Traces the halo-model Compton-y field (`COMPTON_Y`), so its pairings are
    integrated against the halo-model spectra P_my, P_gy and P_yy.
    """

    field = COMPTON_Y

    def __init__(self, tracer: "tSZTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z):
        """Return the Compton-y kernel, shape ``(1, len(z))``."""
        return self._tracer.get_window(z)


class tSZTracer:
    """Class for the kernel of the tSZ Compton-y parameter."""

    def __init__(
        self,
        perturbations: Perturbations,
        z: np.ndarray,
        *,
        beam_fwhm_arcmin: float = 0.0,
    ):
        r"""
        Initialize the class instance.

        Parameters
        ----------
        perturbations : object
            A `GODMAXNonLinearPerturbations` object built with `model_tSZ=True`.
        z : np.ndarray
            A 1-dimensional array used to perform line-of-sight integration.
        beam_fwhm_arcmin : float
            Keyword-only. FWHM of the Gaussian beam of the y map, in arcmin.
            Each y leg of a spectrum is multiplied by `beam(ells)`; 0 (default)
            means no beam. GODMAX's default is 1.4.
        """
        check_nonzero_redshifts(z)
        check_halo_model_field(perturbations, COMPTON_Y, "tSZTracer")
        if beam_fwhm_arcmin < 0:
            raise ValueError("beam_fwhm_arcmin must be non-negative.")
        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.z = z
        self.n_z_bins = 1
        # No spin-2 prefactor (that is for shear).
        self.prefact_toggle = 0
        self.beam_fwhm_arcmin = beam_fwhm_arcmin
        self.compton_y = ComptonYContribution(self)

    def get_contributions(self):
        """Return this tracer's window as its separable Contribution terms.

        Returns:
          contributions (tuple): `(self.compton_y,)`.
        """
        return (self.compton_y,)

    def beam(self, ells):
        r"""Gaussian beam of the y map.

        $$
            B_\ell = \exp\left[-\ell(\ell+1)\,\sigma_b^2/2\right], \qquad
            \sigma_b = \frac{\theta_{\rm FWHM}}{\sqrt{8 \ln 2}}
        $$
        """
        sigma = np.deg2rad(self.beam_fwhm_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
        return np.exp(-ells * (ells + 1.0) * sigma**2 / 2.0)

    def ell_factor(self, ells):
        """Multiplicative ell factor of this tracer's leg (`AngularTwoPoint`)."""
        return self.beam(ells)

    def get_window(self, z):
        r"""Compute the Compton-y window function

        $$
            W^{y}(z) = \frac{1}{1 + z}
        $$

        The pressure profile and its normalisation are inside the halo-model
        Compton-y power spectra.

        Parameters
        ----------
        z: float
            Redshift at which window kernel is being evaluated

        Returns
        -------
        window: np.ndarray
        """
        return np.expand_dims(1.0 / (1.0 + z), 0)

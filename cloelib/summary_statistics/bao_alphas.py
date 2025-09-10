"""Module to compute alpha parameters for the BAO analysis."""

# cloelib imports
from cloelib.cosmology.cosmology import Background
from cloelib.summary_statistics.APDistortion import APDistortion

# General imports
from typing import Union, TypeVar
import numpy as np  # type: ignore
import jax.numpy as jnp

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


class BaryonAcousticOscillations:
    """Class to compute alpha parameters for the BAO analysis."""

    def __init__(
        self,
        background: Background,
        background_fiducial: Background,
        redshifts: np.ndarray,
    ) -> None:
        """Initialize the class instance.

        Parameters
        ----------
        background: Background
            Object following the Background protocol; used to compute background
            quantities needed for the alphas in a given cosmology
        background_fiducial: Background
            Object following the Background protocol; used to compute background
            quantities in the fiducial cosmology
        redshifts: np.ndarray
            Array of redshifts at which the alphas are output
        """
        self.background = background
        self.background_fiducial = background_fiducial
        self.zs = redshifts
        self.Neff = 3.046  # This is hardcoded but should come from background
        self.rd_ratio = self.sound_horizon_drag(
            self.background_fiducial
        ) / self.sound_horizon_drag(self.background)
        self.ap_distortion = APDistortion(background, background_fiducial)

        # the `alphas_dict` attribute is a dictionary containing
        # values of the alphas given background and
        # background_fiducial. The structure follows the hierarchy
        # alphas_dict = {redshift: {alphas: value for alphas} for
        # redshift in zs}, to match the preliminary datamodel for BAO,
        # will need to be reassessed when we freeze the datamodel
        self.alphas_dict = self.set_alphas()

    def alpha_par(self, zs: T) -> T:
        r"""Alpha_parallel.

        Dilation parameter along the line of sight
        ..math::
            \alpha_\parallel(z) &= \frac{H_{\rm fid}(z)}{H(z)} \frac{r_{\rm d,fid}}{r_{\rm d}}
        Parameters
        ----------
        zs: np.array
            redshift

        Returns
        -------
        alpha_par: np.array
            alpha_parallel at requested redshifts
        """
        alpha_par = self.rd_ratio * self.ap_distortion.q_AP_lo(zs)
        return alpha_par

    def alpha_perp(self, zs: T) -> T:
        r"""Alpha_perpendicular.

        Dilation parameter perpendicular to the line of sight
        ..math::
            \alpha_\perp(z) &= \frac{D_{\rm A}(z)}{D_{\rm A, fid}(z)} \frac{r_{\rm d,fid}}{r_{\rm d}}
        Parameters
        ----------
        zs: np.array
            redshift

        Returns
        -------
        alpha_perp: np.array
            alpha_perpendicular at requested redshifts
        """
        alpha_perp = self.rd_ratio * self.ap_distortion.q_AP_tr(zs)
        return alpha_perp

    def alpha_iso(self, alpha_par: T, alpha_perp: T) -> T:
        r"""Alpha_iso.

        Geometrical mean of alpha_parallel and alpha_perpendicular
        ..math::
            \alpha_{\rm iso} = (\alpha_\parallel * \alpha_\perp)^{2/3}

        Parameters
        ----------
        alpha_par: np.ndarray
            alpha_parallel values
        alpha_perp: np.ndarray
            alpha_perpendicular values

        Returns
        -------
        alpha_iso: np.array
            alpha_iso computed from alpha_par, alpha_perp
        """
        return (alpha_par * alpha_perp**2) ** (1 / 3)

    def alpha_AP(self, alpha_par: np.ndarray, alpha_perp: np.ndarray) -> np.ndarray:
        r"""Alpha_AP.

        Ratio of alpha_parallel and alpha_perpendicular
        ..math::
            \alpha_{\rm AP} = (\alpha_\parallel * \alpha_\perp)^{2/3}

        Parameters
        ----------
        alpha_par: np.ndarray
            alpha_parallel values
        alpha_perp: np.ndarray
            alpha_perpendicular values

        Returns
        -------
        alpha_AP: np.array
            alpha_AP computed from alpha_par, alpha_perp
        """
        return alpha_par / alpha_perp

    def set_alphas(self) -> dict:
        r"""Construct a dictionary with values for the BAO alphas.

        The alphas are alpha_par, alpha_perp, alpha_iso and alpha_AP.
        The alphas are evaluated at the redshifts requested at initialisation.

        Returns
        -------
        alphas: dict
            Dictionary containing values for the BAO alphas
        """
        alpha_par = self.alpha_par(self.zs)
        alpha_perp = self.alpha_perp(self.zs)

        alphas = {
            z: {
                "alpha_par": alpha_par[i],
                "alpha_perp": alpha_perp[i],
                "alpha_iso": self.alpha_iso(alpha_par[i], alpha_perp[i]),
                "alpha_AP": self.alpha_AP(alpha_par[i], alpha_perp[i]),
            }
            for i, z in enumerate(self.zs)
        }
        return alphas

    def sound_horizon_drag(self, background):
        r"""Compute the sound horizon at drag epoch.

        Uses the fitting formula Eq.17
        of [1411.1074](https://arxiv.org/abs/1411.1074)

        Parameters
        ----------
        background: Background
            Background class containing cosmology

        Returns
        -------
        r_d: float
            Sound horizon at drag epoch
        """
        omega_cb = background.Omega_cdm0 * background.h**2
        omega_b = background.Omega_b0 * background.h**2
        omega_nu = background.mnu * 93.14

        r_d = (
            56.067
            * np.exp(-49.7 * (omega_nu + 0.002) ** 2)
            / (omega_cb**0.2436 * omega_b**0.128876 * (1 + (self.Neff - 3.046) / 30.6))
        )
        return r_d

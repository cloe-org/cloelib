"""
This module provides Cosmopower-based emulators for linear and nonlinear matter power spectra in various cosmological models.
"""

from cloelib.cosmology.cosmology import Background
from cloelib.auxiliary.extrapolator import extend_spectra

import numpy as np
from scipy import interpolate
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cosmopower as cp  # noqa: E402


class mochiCLASSEmuLinearPerturbations:
    """
    Emulator for the linear matter power spectrum, trained on mochi CLASS, using Cosmopower
    based on the stable basis parametrisation.
    """

    def __init__(
        self, background: Background, redshifts: np.ndarray, cp_file: str = None
    ):
        """
        Initialize the emulator with a given cosmological background and redshift array.

        Parameters
        ----------
        background : Background
            Background cosmology object, providing all necessary cosmological parameters.
        redshifts : np.ndarray
            Array of redshift values for which the power spectrum should be computed.

        Raises
        ------
        AssertionError
            If the geometry is not flat (Omega_k0 != 0).
        ValueError
            If any parameter lies outside the bounds supported by the emulator.
        """

        self.cp_LIN = cp.cosmopower_NN(restore=True, restore_filename=cp_file)

        self.k_emu = np.logspace(-3, 1, 300)  # hard-coded for now
        h = float(background.h)
        self.k_emu *= h

        self.k_min = self.k_emu[0]
        self.k_max = self.k_emu[-1]

        assert background.Omega_k0 == 0, "Non flat geometries not supported"
        assert background.b == 1, "emulator only trained on b=1 for now"

        redshift_max = 5
        self.z = redshifts[redshifts <= redshift_max]

        # Currently emulator implemented for stable basis parametrisation,
        # NOT for the standard mochi_class input of c_s^2, Delta_Mpl etc.
        cp_bounds = {
            "s": np.array([-0.3, 0.3]),
            "a0": np.array([-1, 1]),
            "a1": np.array([-1, 0]),
            "w0": np.array([-1.5, -0.5]),
            "wa": np.array([-0.5, 0.5]),
            "z": np.array([0.0, 5.0]),
        }

        self.params = {
            "s": background.s,
            "a0": background.a0,
            "a1": background.a1,
            "w0": background.w0,
            "wa": background.wa,
        }

        # CURRENTLY TRAINED WITH FIXED COSMOLOGY (PLANCK 2018):
        # fixed_Omega_m = 0.315
        # fixed_Omega_b = 0.049
        # fixed_H0 = 67.4
        # fixed_ns = 0.9649
        # fixed_As = 2.1e-9
        # fixed N_mnu = 0
        # fixed mnu = 0

        for key in self.params.keys():
            if np.prod(self.params[key] - cp_bounds[key]) > 0:
                raise ValueError("Cosmopower emulator out of range.")
            else:
                self.params[key] = np.tile(self.params[key], len(redshifts))

        self.params["z"] = redshifts

        Pk_lin = self.cp_LIN.ten_to_predictions_np(self.params)

        k_out, z_out, Pk_out = extend_spectra(
            self.k_emu,
            self.z,
            Pk_lin * h**-3,
            flag_range=True,
            option_wavenumber="logk2",
            option_redshift="power_law",
            extrap_z=redshifts,
            option_cosmo="const",
            ns=background.ns,
        )

        self.k = k_out
        self.z = z_out
        self.Pk = Pk_out

        pk_int = interpolate.RectBivariateSpline(self.z, self.k, Pk_out, kx=1, ky=1)
        self.Pk_int = pk_int
        # TODO: implement sigma8 and fsigma8 predictions
        # sigma_predictions = self.cp_SIGMA.predictions_np(self.sigma_params)
        # self.sigma8 = sigma_predictions[:, 0]
        # self.fsigma8 = sigma_predictions[:, 1]

    def matter_power_spectrum(self, zs, ks):
        """Compute the linear matter power spectrum P(k, z).

        Parameters
        ----------
        zs : np.ndarray
            Redshifts at which to evaluate the power spectrum.
        ks : np.ndarray
            Wavenumbers in units of Mpc^-1.

        Returns
        -------
        np.ndarray
            Linear matter power spectrum in Mpc^3.
        """
        return self.Pk_int(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""
        Calculate the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.


        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth factor.
        ks : array_like
            Wavenumbers at which to calculate the growth factor.

        Returns:
        --------
        np.ndarray
            The growth factor as a function of redshift and wavenumber.

        """
        if hasattr(self, "Pk_int") and self.Pk_int is not None:
            D_z_k = np.sqrt(self.Pk_int(zs, ks) / self.Pk_int(0, ks))

        return D_z_k

    # TODO: implement growth rate and sigma8 predictions
    # def growth_rate(self) -> np.ndarray:
    #     """
    #     Calculate the growth rate f(z) = d ln D / d ln a.

    #     This is computed as f = fsigma8 / sigma8.

    #     Returns
    #     -------
    #     np.ndarray
    #         The growth rate as a function of redshift.
    #     """
    #     return self.fsigma8 / self.sigma8

    # def sigma8_0(self) -> float:
    #     """
    #     Calculate the sigma8 value.

    #     Returns:
    #     --------
    #     float
    #         The sigma8 value.
    #     """
    #     return self.sigma8[0]

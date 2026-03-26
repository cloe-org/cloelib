"""
Implementation of baryon correction of the matter power spectrum from FlamingoBaryonResponseEmulator on top of CAMB.
"""

import numpy as np
from typing import Optional

from cloelib.cosmology.cosmology import Background

try:
    import camb
    from camb import model
except ImportError as e:
    raise ImportError("camb could not be imported.") from e

try:
    import FlamingoBaryonResponseEmulator as fre
except ImportError:
    raise ImportError("FlamingoBaryonResponseEmulator could not be imported.")


class BaryonBoostMixin:
    """Mixin to add baryonic feedback corrections to nonlinear perturbation classes."""

    def __init__(
        self,
        fgas_sigma: float,
        Mstar_sigma: float,
        jet_fraction: float,
        *args,
        **kwargs,
    ):
        """Initialize baryonic correction parameters.

        Args:
            fgas_sigma: Offset in sigma for gas fraction.
            Mstar_sigma: Offset in sigma for stellar mass function.
            jet_fraction: Fraction of AGN energy in jets (0-1).
        """
        super().__init__(*args, **kwargs)
        self.flamingo_emulator = fre.FlamingoBaryonResponseEmulator()
        self.fgas = fgas_sigma
        self.Mstar = Mstar_sigma
        self.jet = jet_fraction

    def baryonic_suppression(
        self, zs: np.ndarray, ks: np.ndarray, k_hunit: bool = False
    ) -> np.ndarray:
        """Return the predicted baryonic response."""
        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)
        response = np.ones((zs.size, ks.size))

        k_convert = (ks / self.background.H0 * 100) if not k_hunit else ks

        for i in range(len(zs)):
            if zs[i] <= 3.0:
                response[i, :] = self.flamingo_emulator.predict(
                    k_convert, zs[i], self.fgas, self.Mstar, self.jet
                )
                response[i, k_convert > 10**1.5] = self.flamingo_emulator.predict(
                    10**1.5, zs[i], self.fgas, self.Mstar, self.jet
                )
            else:
                response[i, :] = 1.0

        return response

    def baryonic_suppression_with_variance(
        self, zs: np.ndarray, ks: np.ndarray, k_hunit: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return baryonic response and variance."""
        zs = np.atleast_1d(zs)
        ks = np.atleast_1d(ks)
        response = np.ones((zs.size, ks.size))
        variance = np.ones((zs.size, ks.size))

        k_convert = (ks / self.background.H0 * 100) if not k_hunit else ks

        for i in range(len(zs)):
            if zs[i] <= 3.0:
                response[i, :], variance[i, :] = (
                    self.flamingo_emulator.predict_with_variance(
                        k_convert, zs[i], self.fgas, self.Mstar, self.jet
                    )
                )
                response[i, k_convert > 10**1.5], variance[i, k_convert > 10**1.5] = (
                    self.flamingo_emulator.predict_with_variance(
                        10**1.5, zs[i], self.fgas, self.Mstar, self.jet
                    )
                )
            else:
                response[i, :] = 1.0

        return response, variance


class CAMBNonLinearFLAMINGOPerturbations(BaryonBoostMixin):
    """Nonlinear CAMB perturbations with FLAMINGO baryonic feedback correction."""

    def __init__(
        self,
        background: Background,
        redshifts: np.ndarray,
        fgas_sigma: float,
        Mstar_sigma: float,
        jet_fraction: float,
        nonlinear_model: Optional[str] = None,
    ) -> None:
        """Initialize with baryonic boost capabilities."""
        super().__init__(
            fgas_sigma=fgas_sigma,
            Mstar_sigma=Mstar_sigma,
            jet_fraction=jet_fraction,
            background=background,
        )

        self.background = background
        self.kmax = 500
        self.z = redshifts

        # Configure CAMB for nonlinear calculations
        self.background.interface_args["CAMBparams"].NonLinear = model.NonLinear_both
        self.background.interface_args["CAMBparams"].WantCls = False
        self.background.interface_args["CAMBparams"].DoLensing = False
        self.background.interface_args["CAMBparams"].Want_CMB = False
        self.background.interface_args["CAMBparams"].Want_CMB_lensing = False
        self.background.interface_args["CAMBparams"].Want_cl_2D_array = False
        self.background.interface_args["CAMBparams"].WantTransfer = True

        if nonlinear_model:
            self.background.interface_args["CAMBparams"].NonLinearModel.set_params(
                halofit_version=nonlinear_model
            )

        self.background.interface_args["CAMBparams"].set_matter_power(
            redshifts=redshifts, kmax=self.kmax
        )

        self.results = camb.get_results(self.background.interface_args["CAMBparams"])
        self.k, _, self.Pk = self.results.get_nonlinear_matter_power_spectrum(
            hubble_units=False, k_hunit=False
        )

    def matter_power_spectrum(
        self,
        zs: np.ndarray,
        ks: np.ndarray,
        hubble_units: bool = False,
        k_hunit: bool = False,
    ) -> np.ndarray:
        """Compute nonlinear matter power spectrum with baryonic correction."""
        pk_values = self.results.get_matter_power_interpolator(
            nonlinear=True,
            extrap_kmax=self.kmax,
            hubble_units=hubble_units,
            k_hunit=k_hunit,
            var1="delta_tot",
            var2="delta_tot",
        ).P(zs, ks)
        flamingo_correction = self.baryonic_suppression(zs, ks, k_hunit=k_hunit)
        return pk_values * flamingo_correction

    def growth_rate(self) -> np.ndarray:
        """Calculate growth rate."""
        f_z = self.results.get_fsigma8() / self.results.get_sigma8()
        return f_z[::-1]

    def growth_factor(self, zs: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """Calculate the growth factor D(z, k)."""
        D_z_k = np.sqrt(
            self.matter_power_spectrum(zs, ks) / self.matter_power_spectrum(0.0, ks)
        )
        return D_z_k

"""
Implementation of baryon correction of the matter power spectrum from FlamingoBaryonResponseEmulator on top of CAMB.

## Design

**``FlamingoBaryonBoostMixin``** — concrete subclass of the abstract
``BaryonBoostMixin`` (defined in ``cosmology.py``).  Provides
``baryonic_suppression`` and the FLAMINGO-specific
``baryonic_suppression_with_variance`` by calling the external
``FlamingoBaryonResponseEmulator``.  Backend-agnostic: does not import CAMB.

**``CAMBNonLinearFLAMINGOPerturbations``** inherits from both:

- ``FlamingoBaryonBoostMixin`` — baryonic correction via FRE.
- ``CAMBNonLinearPerturbations`` — all CAMB setup and methods
  (``matter_power_spectrum``, ``growth_rate``, ``growth_factor``,
  ``sigma8_0``, …).

``matter_power_spectrum`` is overridden to multiply the CAMB result by the
FLAMINGO suppression factor.  All other CAMB methods are inherited unchanged.

Other backends (BACCOemu, HMcode2020emu) bake the baryonic correction into
their emulator calls rather than factoring it out as a ratio, so they do not
use this mixin.  They can still satisfy ``BaryonicPerturbations`` by
implementing ``baryonic_suppression`` directly.

## Notes

- Adapted from FlamingoBaryonResponseEmulator/flamingo_response_emulator.py
- Adapted from cloelib/cloelib/cosmology/camb_cosmology.py
"""

# General imports
import numpy as np
from typing import Optional

# Cosmology imports
from cloelib.cosmology.cosmology import BaryonBoostMixin
from cloelib.cosmology.camb_cosmology import CAMBNonLinearPerturbations

try:
    import FlamingoBaryonResponseEmulator as fre
except ImportError:
    raise ImportError("FlamingoBaryonResponseEmulator could not be imported.")


# FRE emulator training limits
_K_MAX_TRAINED = 10**1.5  # h/Mpc
_Z_MAX_TRAINED = 3.0


class FlamingoBaryonBoostMixin(BaryonBoostMixin):
    """Concrete baryonic-boost mixin for the FlamingoBaryonResponseEmulator.

    Implements ``baryonic_suppression`` and adds
    ``baryonic_suppression_with_variance`` (FLAMINGO-specific; not part of the
    ``BaryonicPerturbations`` Protocol, but available on any class that uses
    this mixin).

    Host-class contract (attributes set in ``__init__``)
    -----------------------------------------------------
    flamingo_emulator:
        Initialised ``FlamingoBaryonResponseEmulator`` instance.
    fgas : float
        ``fgas_sigma`` — gas fraction offset in sigma. Training range [-8, +2]
        for ``jet=0``; [-4, 0] for ``jet=1``.
    Mstar : float
        ``Mstar_sigma`` — stellar mass function offset in sigma. Range [-1, 0].
    jet : float
        ``jet_fraction`` — AGN jet energy fraction (0 = thermal, 1 = jets).
    background:
        Background-protocol object; ``background.H0`` used for unit conversion.
    """

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _k_to_hMpc(self, ks: np.ndarray, k_hunit: bool) -> np.ndarray:
        """Return ks in h/Mpc (convert from 1/Mpc when k_hunit is False)."""
        if k_hunit:
            return ks
        return ks / self.background.H0 * 100

    # ------------------------------------------------------------------
    # BaryonBoostMixin implementation
    # ------------------------------------------------------------------

    def baryonic_suppression(
        self, zs: np.ndarray, ks: np.ndarray, k_hunit: bool = False
    ) -> np.ndarray:
        """Return the FLAMINGO baryonic suppression factor P_hydro/P_DMO.

        For z > 3 (outside FRE training range) the correction is 1.
        For k > 10^1.5 h/Mpc the correction is clamped to its value at the
        maximum trained wavenumber.

        Parameters
        ----------
        zs:
            Redshifts, shape (nz,).
        ks:
            Wavenumbers, shape (nk,).
        k_hunit:
            If ``True`` ks are in h/Mpc; otherwise in 1/Mpc.

        Returns
        -------
        np.ndarray, shape (nz, nk)
        """
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        ks = np.atleast_1d(np.asarray(ks, dtype=float))
        ks_hMpc = self._k_to_hMpc(ks, k_hunit)
        ks_clamped = np.where(ks_hMpc > _K_MAX_TRAINED, _K_MAX_TRAINED, ks_hMpc)

        response = np.ones((zs.size, ks.size))
        for i, z in enumerate(zs):
            if z <= _Z_MAX_TRAINED:
                response[i, :] = self.flamingo_emulator.predict(
                    ks_clamped, z, self.fgas, self.Mstar, self.jet
                )
        return response

    # ------------------------------------------------------------------
    # FLAMINGO-specific extra (not in BaryonicPerturbations Protocol)
    # ------------------------------------------------------------------

    def baryonic_suppression_with_variance(
        self, zs: np.ndarray, ks: np.ndarray, k_hunit: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the FLAMINGO suppression factor and its emulator variance.

        For z > 3 the response is 1 and variance is 0.
        For k > 10^1.5 h/Mpc the values are clamped to the maximum trained k.

        Parameters
        ----------
        zs:
            Redshifts, shape (nz,).
        ks:
            Wavenumbers, shape (nk,).
        k_hunit:
            If ``True`` ks are in h/Mpc; otherwise in 1/Mpc.

        Returns
        -------
        response : np.ndarray, shape (nz, nk)
        variance : np.ndarray, shape (nz, nk)
        """
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        ks = np.atleast_1d(np.asarray(ks, dtype=float))
        ks_hMpc = self._k_to_hMpc(ks, k_hunit)
        ks_clamped = np.where(ks_hMpc > _K_MAX_TRAINED, _K_MAX_TRAINED, ks_hMpc)

        response = np.ones((zs.size, ks.size))
        variance = np.zeros((zs.size, ks.size))
        for i, z in enumerate(zs):
            if z <= _Z_MAX_TRAINED:
                response[i, :], variance[i, :] = (
                    self.flamingo_emulator.predict_with_variance(
                        ks_clamped, z, self.fgas, self.Mstar, self.jet
                    )
                )
        return response, variance


class CAMBNonLinearFLAMINGOPerturbations(
    FlamingoBaryonBoostMixin, CAMBNonLinearPerturbations
):
    """CAMB nonlinear perturbations with FLAMINGO baryonic feedback correction.

    Inherits all CAMB setup and methods from ``CAMBNonLinearPerturbations`` and
    gains baryonic suppression via ``FlamingoBaryonBoostMixin``.  Only
    ``matter_power_spectrum`` is overridden to multiply P(k,z) by the FLAMINGO
    response.

    Parameters
    ----------
    background:
        Background-protocol object (e.g. ``CAMBBackground``).
    redshifts:
        Array of redshifts for CAMB calculations.
    fgas_sigma:
        Offset in sigma from the calibration gas fraction.  Trained range:
        [-8, +2] for ``jet_fraction=0``; [-4, 0] for ``jet_fraction=1``.
    Mstar_sigma:
        Offset in sigma from the stellar mass function calibration.
        Trained range: [-1, 0].
    jet_fraction:
        Fraction of AGN energy in collimated jets (0 = thermal, 1 = jets).
    nonlinear_model:
        Halofit version string (e.g. ``"takahashi"``).  ``None`` uses the CAMB
        default.
    """

    def __init__(
        self,
        background,
        redshifts: np.ndarray,
        fgas_sigma: float,
        Mstar_sigma: float,
        jet_fraction: float,
        nonlinear_model: Optional[str] = None,
    ) -> None:
        # Initialise CAMBNonLinearPerturbations (CAMB setup, get_results, …)
        # NOTE: super().__init__ is intentionally NOT used here.  Protocol's
        # metaclass injects __init__ into BaryonBoostMixin.__dict__, which sits
        # before CAMBNonLinearPerturbations in the MRO and would swallow all
        # arguments without forwarding them.
        CAMBNonLinearPerturbations.__init__(
            self,
            background=background,
            redshifts=redshifts,
            nonlinear_model=nonlinear_model,
        )

        # Attributes consumed by FlamingoBaryonBoostMixin
        self.fgas = fgas_sigma
        self.Mstar = Mstar_sigma
        self.jet = jet_fraction
        self.flamingo_emulator = fre.FlamingoBaryonResponseEmulator()

    # ------------------------------------------------------------------
    # Override: P_CAMB × B_FLAMINGO
    # ------------------------------------------------------------------

    def matter_power_spectrum(
        self, zs, ks, hubble_units=False, k_hunit=False
    ) -> np.ndarray:
        r"""Nonlinear matter power spectrum with FLAMINGO baryonic feedback.

        Parameters
        ----------
        zs: numpy.ndarray
            Redshifts.
        ks: numpy.ndarray
            Wavenumbers.
        hubble_units: bool, optional
            Return P(k) in :math:`({\rm Mpc}/h)^3` units.  Default ``False``.
        k_hunit: bool, optional
            ``ks`` are in h/Mpc.  Default ``False`` (1/Mpc).

        Returns
        -------
        pk: numpy.ndarray, shape (len(zs), len(ks))
            :math:`P_{\rm CAMB}(k,z) \times B_{\rm FLAMINGO}(k,z)`.
        """
        pk_camb = super().matter_power_spectrum(
            zs, ks, hubble_units=hubble_units, k_hunit=k_hunit
        )
        return pk_camb * self.baryonic_suppression(zs, ks, k_hunit=k_hunit)

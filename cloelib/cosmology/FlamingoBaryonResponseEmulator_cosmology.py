"""
Implementation of baryon correction of the matter power spectrum from FlamingoBaryonResponseEmulator on top of CAMB.

## Design

**``FlamingoBaryonBoostMixin``** — concrete subclass of the abstract
``BaryonBoostMixin`` (defined in ``cosmology.py``).  Provides
``baryonic_suppression`` and the FLAMINGO-specific
``baryonic_suppression_with_variance`` by calling the external
``FlamingoBaryonResponseEmulator``.  Backend-agnostic: does not import CAMB.

**``CAMBNonLinearFLAMINGOBaryonicPerturbations``** inherits from both:

- ``FlamingoBaryonBoostMixin`` — baryonic correction via FRE.
- ``CAMBNonLinearPerturbations`` — all CAMB setup and methods
  (``matter_power_spectrum``, ``growth_rate``, ``growth_factor``,
  ``sigma8_0``, …).

## Notes

- Adapted from FlamingoBaryonResponseEmulator/flamingo_response_emulator.py
"""

# General imports
import numpy as np

# Cosmology imports
from cloelib.cosmology.cosmology import BaryonBoostMixin, with_baryon_boost
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

    Implements ``baryonic_suppression`` (and the FLAMINGO-specific
    ``baryonic_suppression_with_variance``).  Can be combined with *any*
    nonlinear perturbations class::

        class MyPert(FlamingoBaryonBoostMixin, SomeNonLinearPerturbations):
            def __init__(self, background, ..., fgas_sigma=0, Mstar_sigma=0, jet_fraction=0):
                SomeNonLinearPerturbations.__init__(self, background, ...)
                FlamingoBaryonBoostMixin.__init__(self, fgas_sigma, Mstar_sigma, jet_fraction)

            def matter_power_spectrum(self, zs, ks, **kwargs):
                return SomeNonLinearPerturbations.matter_power_spectrum(self, zs, ks, **kwargs)\
                    * self.baryonic_suppression(zs, ks, k_hunit=kwargs.get("k_hunit", False))

    Or use the :func:`~cloelib.cosmology.cosmology.with_baryon_boost` factory for
    zero-boilerplate class creation.
    """

    # ------------------------------------------------------------------
    # Initialiser
    # ------------------------------------------------------------------

    def __init__(
        self,
        fgas_sigma: float = 0.0,
        Mstar_sigma: float = 0.0,
        jet_fraction: float = 0.0,
    ) -> None:
        """Initialise the FLAMINGO mixin attributes.

        Call this **after** the base NonLinear perturbations ``__init__``
        so that ``self.background`` is already set.

        Parameters
        ----------
        fgas_sigma:
            Offset from calibration gas fraction, in σ.  Training range
            [-8, +2] for ``jet_fraction=0``; [-4, 0] for ``jet_fraction=1``.
        Mstar_sigma:
            Offset from stellar mass function calibration, in σ.
            Training range [-1, 0].
        jet_fraction:
            AGN jet energy fraction (0 = pure thermal, 1 = pure jets).
        """
        self.fgas = fgas_sigma
        self.Mstar = Mstar_sigma
        self.jet = jet_fraction
        self.flamingo_emulator = fre.FlamingoBaryonResponseEmulator()

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


#: CAMB nonlinear perturbations with FLAMINGO baryonic feedback correction.
#:
#: Created via :func:`~cloelib.cosmology.cosmology.with_baryon_boost`.
#: Accepts all ``CAMBNonLinearPerturbations`` arguments plus a
#: ``baryon_kwargs`` dict forwarded to ``FlamingoBaryonBoostMixin``::
#:
#:     from cloelib.cosmology.FlamingoBaryonResponseEmulator_cosmology import (
#:         CAMBNonLinearFLAMINGOBaryonicPerturbations,
#:     )
#:
#:     pert = CAMBNonLinearFLAMINGOBaryonicPerturbations(
#:         background=bg,
#:         redshifts=zs,
#:         baryon_kwargs=dict(fgas_sigma=0.0, Mstar_sigma=0.0, jet_fraction=0.0),
#:     )
CAMBNonLinearFLAMINGOBaryonicPerturbations = with_baryon_boost(
    CAMBNonLinearPerturbations, FlamingoBaryonBoostMixin
)

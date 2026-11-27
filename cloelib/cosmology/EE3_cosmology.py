"""Implementation of linear and nonlinear Perturbation cosmology using EuclidEmulator3.

EuclidEmulator3 (EE3) emulates the nonlinear matter power
spectrum of PKDGRAV3 N-body simulations in an 8-parameter w0waCDM + massive
neutrino space (``Omega_cdm, Omega_baryon, A_s, ns, hubble, neutrino_mass, w0,
wa``). It ships its own linear-theory, growth and HMcode-2020 emulators.

## Design

**``EE3LinearPerturbations``** -- linear total and cold spectra from the EE3
linear emulator (``emu.lin.pk``) and the scale-dependent growth factors from
the dedicated EE3 growth emulator (``emu.lin.growth``), valid for ``0 <= z <= 9``.

**``EE3NonLinearPerturbations``** -- the EE3 nonlinear cold boost
``B(k, z) = P_cc^NL / P_cc^L`` (``emu.sim.lin_boost``, ``0 <= z <= 6``,
``0.01 <= k <= 12.4 h/Mpc``) applied on top of *any* linear perturbations, in the
same way as ``EE2NonLinearPerturbations``:

* ``P_cb^NL = B * P_cb^L``;
* ``P_mm^NL`` from the linear-neutrino response EE3 itself uses (massive
  neutrinos stay linear and fully correlated with the cold field):
  ``sqrt(P_mm^NL) = (1 - f_nu) sqrt(P_cb^NL) + f_nu sqrt(P_nu^L)``. With this
  response the cb x matter cross spectrum is exactly ``sqrt(P_cb P_mm)``, hence
  ``cb_cross_prescription = CB_CROSS_CORRELATED``.

**``EE3HMcodePerturbations``** -- the HMcode-2020 emulator shipped with EE3
(``emu.hmc.boost``, ``0 <= z <= 9``, ``1e-4 <= k <= 100 h/Mpc``): cold and total
HMcode boosts over linear, applied on top of *any* linear perturbations,
``P_cb^NL = B_hm,cb * P_cb^L`` and ``P_mm^NL = B_hm,mm * P_mm^L`` (HMcode treats
the neutrinos itself). Same cb x matter cross convention as above, which for
HMcode is an approximation.

With ``EE3LinearPerturbations`` as the linear input the nonlinear classes
reproduce EE3 exactly (up to interpolation in k). With another linear backend
(CAMB, CLASS, BACCOemu, ...) the boosts are rescaled onto that backend's linear
spectrum.

All EE3 calls are made once per cosmology with the full array of redshifts
(one network / GP pass for all of them). The spectra are tabulated at the
requested redshifts and interpolated linearly in between.

Outside the emulated ranges the boosts are extended with power laws in ``k`` and
``1 + z`` (``cloelib.auxiliary.extrapolator.extend_spectra``); below
``k = 0.01 h/Mpc`` the EE3 boost is 1, as EE3 anchors itself to linear theory
there.

Cosmologies outside the EE3 prior (or with ``w0 + wa > 0``) raise a
``ValueError`` from the emulator itself.

## Notes

- Requires a ``euclidemulator3`` whose ``emu.lin`` and ``emu.hmc`` accept arrays
  of redshifts (its ``main`` branch, November 2026).
- EE3 assumes three degenerate massive neutrino species with
  ``Omega_nu h^2 = sum(m_nu) / 93.14 eV``; only the total mass is passed on.
- The growth factors are those of the EE3 growth emulator, normalised to 1 at
  ``z = 0``; they agree with ``sqrt(P(z) / P(0))`` of the linear emulator to the
  emulation accuracy, not to rounding.
- The emulator data are downloaded on first use (``python -m
  euclidemulator3._download``). On machines without network access prefetch them
  and/or point ``EE3_DATA_DIR`` at a local copy.
"""

# cloelib imports
from cloelib.cosmology.cosmology import (
    CB_CROSS_CORRELATED,
    Background,
    Perturbations,
    WithWavenumberGrid,
)
from cloelib.auxiliary.extrapolator import extend_spectra

from scipy import interpolate

# General imports
import numpy as np
from typing import Optional, Protocol, Sequence, runtime_checkable

# Cosmology imports
try:
    from euclidemulator3 import (
        EuclidEmulator3,
        neutrino_fraction,
        neutrino_linear_from_total,
        total_matter_response,
    )
    from euclidemulator3.theory import Z_MAX_THEORY

    emu = EuclidEmulator3()
except ImportError:
    raise ImportError("EuclidEmulator3 could not be imported or initialised.")


@runtime_checkable
class LinearPerturbationsWithK(Perturbations, WithWavenumberGrid, Protocol):
    """A `Perturbations` implementation that also exposes its `.k` grid."""


def _ee3_params(background: Background) -> dict:
    """EE3 input parameters (density convention) from a cloelib background."""
    assert background.Omega_k0 == 0, "Non flat geometries not supported"
    return {
        "Omega_cdm": background.Omega_cdm0,
        "Omega_baryon": background.Omega_b0,
        "A_s": background.As,
        "ns": background.ns,
        "hubble": background.H0 / 100.0,
        "neutrino_mass": _set_neutrino_masses(background),
        "w0": background.w0,
        "wa": background.wa,
    }


def _emulated_redshifts(redshifts: np.ndarray, z_max: float) -> np.ndarray:
    """Sorted unique redshifts inside the emulated range [0, z_max]."""
    z = np.unique(np.asarray(redshifts, dtype=float))
    z = z[(z >= 0.0) & (z <= z_max)]
    if z.size == 0:
        raise ValueError(
            f"No redshift inside the EuclidEmulator3 range [0, {z_max:g}]."
        )
    return z


def _interp_loglog_rows(x_src, rows, x_dst) -> np.ndarray:
    """Linear interpolation of ``log(rows)`` in ``log(x)``, row by row, vectorised.

    Constant (clamped) outside ``x_src``. ``rows`` (n, len(x_src)) -> (n, len(x_dst)).
    """
    lx_src = np.log(x_src)
    lx = np.clip(np.log(x_dst), lx_src[0], lx_src[-1])
    i = np.clip(np.searchsorted(lx_src, lx) - 1, 0, lx_src.size - 2)
    w = (lx - lx_src[i]) / (lx_src[i + 1] - lx_src[i])
    lr = np.log(rows)
    return np.exp(lr[:, i] * (1.0 - w) + lr[:, i + 1] * w)


class EE3LinearPerturbations:
    """Linear perturbations from the EuclidEmulator3 linear emulator, compatible with the Perturbations protocol."""

    cb_cross_prescription = CB_CROSS_CORRELATED

    def __init__(self, background: Background, redshifts: np.ndarray):
        """Initialize the EE3LinearPerturbations instance.

        Args:
            background (Background): Background cosmology.
            redshifts (np.ndarray): Redshifts at which the spectra are tabulated.
                Those above the linear emulator range (z = 9) are extrapolated.
        """
        self.emu = emu
        self.background = background
        self.params_emu = _ee3_params(background)
        h = self.background.h

        # All redshifts in one network pass per spectrum.
        z_emu = _emulated_redshifts(redshifts, Z_MAX_THEORY)
        k_h = np.asarray(self.emu.lin.k)
        Pk = self.emu.lin.pk(self.params_emu, z_emu, kind="total")
        Pk_cb = self.emu.lin.pk(self.params_emu, z_emu, kind="cold")

        extend = dict(
            flag_range=True,
            option_wavenumber="logk2",
            option_redshift="power_law",
            extrap_z=np.unique(redshifts),
            option_cosmo="const",
            ns=self.background.ns,
        )
        k_out, z_out, Pk_out = extend_spectra(k_h * h, z_emu, Pk * h**-3, **extend)
        _, _, Pk_cb_out = extend_spectra(k_h * h, z_emu, Pk_cb * h**-3, **extend)

        self.k = k_out
        self.z = z_out
        self.Pk = Pk_out
        self.Pk_cb = Pk_cb_out
        # log P against log k, linear: exactly how emu.sim.pk_nl itself
        # interpolates the emulated linear spectrum (np.interp in log-log).
        self._lnPk_interp = interpolate.RectBivariateSpline(
            self.z, np.log(self.k), np.log(Pk_out), kx=1, ky=1
        )
        self._lnPk_cb_interp = interpolate.RectBivariateSpline(
            self.z, np.log(self.k), np.log(Pk_cb_out), kx=1, ky=1
        )

    def Pk_interp(self, zs, ks) -> np.ndarray:
        """Interpolated total linear P(z, k), k in 1/Mpc, shape (nz, nk)."""
        return np.exp(self._lnPk_interp(np.atleast_1d(zs), np.log(np.atleast_1d(ks))))

    def Pk_cb_interp(self, zs, ks) -> np.ndarray:
        """Interpolated cold linear P_cb(z, k), k in 1/Mpc, shape (nz, nk)."""
        return np.exp(
            self._lnPk_cb_interp(np.atleast_1d(zs), np.log(np.atleast_1d(ks)))
        )

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        r"""Total (cdm + baryons + neutrinos) linear matter power spectrum.

        Args:
            zs (numpy.ndarray): Redshifts.
            ks (numpy.ndarray): Wavenumbers in 1/Mpc.

        Returns:
            np.ndarray: P(z, k) in Mpc^3, shape (nz, nk).
        """
        return self.Pk_interp(zs, ks)

    def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
        r"""Cold (cdm + baryons) linear matter power spectrum.

        Args:
            zs (numpy.ndarray): Redshifts.
            ks (numpy.ndarray): Wavenumbers in 1/Mpc.

        Returns:
            np.ndarray: P_cb(z, k) in Mpc^3, shape (nz, nk).
        """
        return self.Pk_cb_interp(zs, ks)

    def _growth(self, zs, ks, kind: str) -> np.ndarray:
        r"""Scale-dependent growth $D(z, k)$ from the EE3 growth emulator, shape (nz, nk).

        One batched call for all ``zs`` inside the emulator range, log-log
        interpolation in k (constant outside ``1e-4 <= k <= 100 h/Mpc``, where
        the growth is scale independent). Above ``z = 9`` the growth is continued
        with the extrapolated linear spectrum,
        $D(z) = D(9) \sqrt{P(z) / P(9)}$.
        """
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        ks = np.atleast_1d(np.asarray(ks, dtype=float))
        zc = np.minimum(zs, Z_MAX_THEORY)
        z_u, inv = np.unique(zc, return_inverse=True)
        D = np.atleast_2d(self.emu.lin.growth(self.params_emu, z_u, kind=kind))
        out = _interp_loglog_rows(
            np.asarray(self.emu.lin.k), D, ks / self.background.h
        )[inv]
        high = zs > Z_MAX_THEORY
        if np.any(high):
            P = self.Pk_interp if kind == "total" else self.Pk_cb_interp
            out[high] *= np.sqrt(P(zs[high], ks) / P(Z_MAX_THEORY, ks))
        return out

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""Total linear growth factor $D(z, k)$, normalised to 1 at z = 0 (EE3 growth emulator)."""
        return self._growth(zs, ks, "total")

    def growth_factor_cb(self, zs, ks) -> np.ndarray:
        r"""Cold linear growth factor $D_{cb}(z, k)$, normalised to 1 at z = 0 (EE3 growth emulator)."""
        return self._growth(zs, ks, "cold")

    def growth_rate(self, zs=None, ks=None) -> np.ndarray:
        r"""Total linear growth rate $f = d\ln D / d\ln a$.

        Args:
            zs (Optional[array_like]): Redshifts. Defaults to `self.z`.
            ks (array_like): Wavenumbers in 1/Mpc. Required, since with massive
                neutrinos the growth rate is scale dependent.

        Returns:
            np.ndarray: The growth rate, shape (nz, nk).
        """
        if zs is None:
            zs = self.z
        if ks is None:
            raise ValueError(
                "The EE3 growth rate is scale dependent: `ks` must be provided."
            )
        return _growth_rate_from_growth_factor(self.growth_factor, self.z, zs, ks)

    def growth_rate_cb(self, zs, ks) -> np.ndarray:
        r"""Cold linear growth rate $f_{cb} = d\ln D_{cb} / d\ln a$, shape (nz, nk)."""
        return _growth_rate_from_growth_factor(self.growth_factor_cb, self.z, zs, ks)

    def sigma8_0(self) -> float:
        """Total-matter linear sigma8 at z = 0 (exact top-hat quadrature in EE3)."""
        return float(self.emu.lin.sigma8(self.params_emu, 0.0)[1])

    def sigma8_0_cb(self) -> float:
        """Cold-matter linear sigma8 at z = 0 (exact top-hat quadrature in EE3)."""
        return float(self.emu.lin.sigma8(self.params_emu, 0.0)[0])


class _EE3BoostedPerturbations:
    """Common machinery of the nonlinear EE3 classes: a cold boost tabulated at
    the requested redshifts, extended outside the emulated range and applied to
    the linear perturbations. Subclasses provide ``matter_power_spectrum`` and
    declare their ``cb_cross_prescription``."""

    # Set by the subclasses from `_boost_table`.
    z: np.ndarray
    boost_interp: interpolate.RectBivariateSpline
    _boost_k_range: tuple[float, float]

    def __init__(
        self,
        background: Background,
        linearperturbations: LinearPerturbationsWithK,
        redshifts: np.ndarray,
    ):
        self.background = background
        self.linearperturbations = linearperturbations
        self.params_emu = _ee3_params(background)
        self.f_nu = float(
            neutrino_fraction(
                self.params_emu["Omega_cdm"],
                self.params_emu["Omega_baryon"],
                self.params_emu["neutrino_mass"],
                self.params_emu["hubble"],
            )
        )
        # Output k grid taken from the linear perturbations (as for EE2), whose
        # sampling is the one used later in the C_ell integrals.
        self.k = linearperturbations.k

    def _boost_table(self, k_hmpc, z_nodes, boost, redshifts):
        """Extend a boost tabulated on (z_nodes, k_hmpc [h/Mpc]) and return
        ``(z_out, interpolator)``, the interpolator being linear in (z, ln k) of ln B."""
        k_emu = np.asarray(k_hmpc) * self.background.h  # h/Mpc -> 1/Mpc
        k_out, z_out, boost_out = extend_spectra(
            k_emu,
            z_nodes,
            np.atleast_2d(boost),
            flag_range=True,
            option_wavenumber="power_law",
            option_redshift="power_law",
            extrap_z=np.unique(redshifts),
            option_cosmo="const",
            ns=self.background.ns,
        )
        # log B against log k: B > 0 everywhere and the power-law extension is
        # linear in these variables.
        interp = interpolate.RectBivariateSpline(
            z_out, np.log(k_out), np.log(boost_out), kx=1, ky=1
        )
        return z_out, interp, (float(k_emu[0]), float(k_emu[-1]))

    @property
    def boost_k_range(self) -> tuple[float, float]:
        """Wavenumbers in 1/Mpc over which the boost is emulated (extrapolated outside)."""
        return self._boost_k_range

    def nonlinear_boost(self, zs, ks) -> np.ndarray:
        """Return the cold boost B(z, k) = P_cb^NL / P_cb^L.

        Args:
            zs (array_like): Redshifts.
            ks (array_like): Wavenumbers in 1/Mpc.

        Returns:
            np.ndarray: Boost, shape (nz, nk).
        """
        return np.exp(self.boost_interp(np.atleast_1d(zs), np.log(np.atleast_1d(ks))))

    def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
        r"""Nonlinear cold (cdm + baryons) power spectrum, $B \, P_{cb}^L$.

        Args:
            zs (numpy.ndarray): Redshifts.
            ks (numpy.ndarray): Wavenumbers in 1/Mpc.

        Returns:
            np.ndarray: P_cb(z, k) in Mpc^3, shape (nz, nk).
        """
        return self.nonlinear_boost(
            zs, ks
        ) * self.linearperturbations.matter_power_spectrum_cb(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        """Total linear growth factor, from the linear perturbations."""
        return self.linearperturbations.growth_factor(zs, ks)

    def growth_rate(
        self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Total linear growth rate, from the linear perturbations.

        Args:
            zs (Optional[np.ndarray]): Redshifts. Defaults to `self.z`.
            ks (Optional[np.ndarray]): Wavenumbers in 1/Mpc.

        Returns:
            np.ndarray: The linear growth rate.
        """
        return self.linearperturbations.growth_rate(self.z if zs is None else zs, ks)

    def sigma8_0(self) -> float:
        """Total-matter linear sigma8 at z = 0, from the linear perturbations."""
        return self.linearperturbations.sigma8_0()


class EE3NonLinearPerturbations(_EE3BoostedPerturbations):
    """Nonlinear perturbations using the EuclidEmulator3 boost, compatible with the Perturbations protocol."""

    cb_cross_prescription = CB_CROSS_CORRELATED

    def __init__(
        self,
        background: Background,
        linearperturbations: LinearPerturbationsWithK,
        redshifts: np.ndarray,
    ):
        """Initialize the EE3NonLinearPerturbations instance.

        Args:
            background (Background): Background cosmology.
            linearperturbations (LinearPerturbationsWithK): Linear perturbations the
                EE3 cold boost is applied to. ``EE3LinearPerturbations`` gives the
                full EE3 prediction from the parameters alone.
            redshifts (np.ndarray): Redshifts. The boost is emulated at all of
                them in one call; those above the boost range (z = 6) are
                extrapolated with a power law in 1 + z.
        """
        super().__init__(background, linearperturbations, redshifts)
        z_emu = _emulated_redshifts(redshifts, emu.z_max)
        # Standard boost B = P_cc^NL / P_cc^L on the native EE3 grid (h/Mpc),
        # from the parameters alone (internal EE3 linear spectrum).
        boost = emu.sim.lin_boost(self.params_emu, z_emu)
        self.z, self.boost_interp, self._boost_k_range = self._boost_table(
            emu.k, z_emu, boost, redshifts
        )

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        r"""Nonlinear total-matter power spectrum.

        Massive neutrinos are linear and fully correlated with the cold field
        (the response EE3 uses itself):
        $\sqrt{P_{mm}} = (1 - f_\nu)\sqrt{P_{cb}^{NL}} + f_\nu \sqrt{P_\nu^L}$.

        Args:
            zs (numpy.ndarray): Redshifts.
            ks (numpy.ndarray): Wavenumbers in 1/Mpc.

        Returns:
            np.ndarray: P(z, k) in Mpc^3, shape (nz, nk).
        """
        boost = self.nonlinear_boost(zs, ks)
        Pcb_L = self.linearperturbations.matter_power_spectrum_cb(zs, ks)
        if self.f_nu == 0.0:
            # Without massive neutrinos total and cold matter coincide; using
            # P_cb^L also for the total keeps the two exactly equal, as in EE3.
            return boost * Pcb_L
        Pmm_L = self.linearperturbations.matter_power_spectrum(zs, ks)
        Pnu_L = neutrino_linear_from_total(Pcb_L, Pmm_L, self.f_nu)
        return total_matter_response(boost * Pcb_L, Pnu_L, self.f_nu)

    def boost_error(self, zs, ks, level: str = "err_2sigma") -> np.ndarray:
        """Calibrated relative error of the EE3 boost (``emu.sim.lin_boost_err``).

        Useful for a theory-error covariance. Evaluated on demand (not at
        construction), in one call for all redshifts inside the emulated range;
        zero elsewhere.

        Args:
            zs (array_like): Redshifts.
            ks (array_like): Wavenumbers in 1/Mpc. Clamped to the emulated range.
            level (str): ``"err_median"``, ``"err_1sigma"``, ``"err_2sigma"`` or
                ``"err_3sigma"``.

        Returns:
            np.ndarray: Relative error, shape (nz, nk).
        """
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        ks = np.atleast_1d(np.asarray(ks, dtype=float))
        out = np.zeros((zs.size, ks.size))
        inside = (zs >= 0.0) & (zs <= emu.z_max)
        if np.any(inside):
            err = emu.sim.lin_boost_err(self.params_emu, zs[inside])[level]
            out[inside] = _interp_loglog_rows(
                np.asarray(emu.k), np.atleast_2d(err), ks / self.background.h
            )
        return out


class EE3HMcodePerturbations(_EE3BoostedPerturbations):
    """Nonlinear perturbations from the EE3 HMcode-2020 emulator, compatible with the Perturbations protocol."""

    cb_cross_prescription = CB_CROSS_CORRELATED

    def __init__(
        self,
        background: Background,
        linearperturbations: LinearPerturbationsWithK,
        redshifts: np.ndarray,
    ):
        """Initialize the EE3HMcodePerturbations instance.

        Args:
            background (Background): Background cosmology.
            linearperturbations (LinearPerturbationsWithK): Linear perturbations the
                HMcode boosts are applied to. ``EE3LinearPerturbations`` gives the
                emulated HMcode-2020 spectra from the parameters alone.
            redshifts (np.ndarray): Redshifts. The boosts are emulated at all of
                them in one call; those above the emulator range (z = 9) are
                extrapolated with a power law in 1 + z.
        """
        super().__init__(background, linearperturbations, redshifts)
        z_emu = _emulated_redshifts(redshifts, Z_MAX_THEORY)
        k_h = np.asarray(emu.hmc.k)
        # B_hm = P_HMcode / P_lin, cold and total, on the native theory grid (h/Mpc).
        boost_cb = emu.hmc.boost(self.params_emu, z_emu, kind="cold")
        self.z, self.boost_interp, self._boost_k_range = self._boost_table(
            k_h, z_emu, boost_cb, redshifts
        )
        if self.f_nu == 0.0:
            self.boost_mm_interp = self.boost_interp
        else:
            boost_mm = emu.hmc.boost(self.params_emu, z_emu, kind="total")
            _, self.boost_mm_interp, _ = self._boost_table(
                k_h, z_emu, boost_mm, redshifts
            )

    def nonlinear_boost_mm(self, zs, ks) -> np.ndarray:
        """Return the total-matter HMcode boost B_mm(z, k) = P_mm^HM / P_mm^L, shape (nz, nk)."""
        return np.exp(
            self.boost_mm_interp(np.atleast_1d(zs), np.log(np.atleast_1d(ks)))
        )

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        r"""Nonlinear total-matter power spectrum, $B_{mm} \, P_{mm}^L$.

        Args:
            zs (numpy.ndarray): Redshifts.
            ks (numpy.ndarray): Wavenumbers in 1/Mpc.

        Returns:
            np.ndarray: P(z, k) in Mpc^3, shape (nz, nk).
        """
        if self.f_nu == 0.0:
            # Cold and total coincide; keep them exactly equal (the cold and
            # total HMcode networks are trained separately).
            return self.matter_power_spectrum_cb(zs, ks)
        return self.nonlinear_boost_mm(
            zs, ks
        ) * self.linearperturbations.matter_power_spectrum(zs, ks)


def _set_neutrino_masses(background: Background) -> float:
    """Total neutrino mass in eV, the only neutrino input EE3 takes.

    EE3 is trained on three degenerate massive species with N_eff = 3.044, so the
    background must use the same number of relativistic species.
    """
    if not np.isclose(background.N_eff, 3.044, rtol=1e-3):
        raise ValueError(
            "EE3 only supports N_eff=3.044 (three degenerate massive neutrinos). "
            f"Found N_eff={background.N_eff}."
        )
    if isinstance(background.mnu, (Sequence, np.ndarray)):
        return float(np.sum(np.asarray(background.mnu)))
    return float(background.mnu)


def _growth_rate_from_growth_factor(growth_factor, z_grid, zs, ks) -> np.ndarray:
    r"""Compute $f = d\ln D / d\ln a$ from a growth factor $D(z, k)$.

    Cubic spline in $\ln a$ over the perturbations' own redshift grid, evaluated
    at `zs` (same scheme as the BACCOemu backend).
    """
    z_sorted = np.sort(np.asarray(z_grid))
    growth = growth_factor(z_sorted, ks)
    cs = interpolate.CubicSpline(
        np.log(1 / (1 + z_sorted))[::-1], np.log(growth)[::-1], axis=0
    )
    return cs(np.log(1 / (1 + np.atleast_1d(zs))), 1)

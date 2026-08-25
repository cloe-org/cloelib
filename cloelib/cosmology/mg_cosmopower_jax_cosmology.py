"""Binned modified-gravity perturbations using CosmoPower-JAX boost emulators.

This module applies a per-bin modified-gravity *boost*

    B(k, z) = P_MG(k, z) / P_LCDM(k, z)

predicted by Ivan's CosmoPower-JAX emulators on top of an external LCDM baseline
(another cloelib ``Perturbations`` object). The boost is applied as a
multiplicative operator *at query time* on the baseline's own k-grid:

    P_MG(k, z) = B(k, z) * P_LCDM(k, z)

so that at B = 1 (mu = eta = 1, GR) the result is the baseline LCDM spectrum
*exactly* -- the GR limit is recovered to machine precision, with no regridding
artefacts.

Drop-in replacement for the ``LinPerturbations`` / ``NonLinPerturbations`` classes
in ``cloelike`` (``EuclidLikelihood_photo_Cls``). Provides the modified lensing
parameter ``Sigma(z) = mu(1 + eta)/2`` that ``photo.py`` applies to the WL kernel.

Emulators (per MG bin, bin_index dropped), trained on log10(boost):
    mg-boost-linear-bin{0..4}.npz     inputs [Omega_m,Omega_b,h,ns,lnAs,mu,eta,z]  (800 k)
    mg-boost-nonlinear-bin{0..4}.npz  inputs [Omega_m,Omega_b,h,ns,lnAs,mu,z]       (1024 k)
  k in h/Mpc; lnAs = ln(1e10 * As); loaded with probe='custom_log' so predict()
  returns the boost directly. eta has NO nonlinear effect (not an NL input).

Injecting mu/eta/bin_index
--------------------------
cloelike builds ``background`` from a fixed cosmo-key list (no mu/eta/bin_index)
and calls ``LinPerturbations(background, zs)`` / ``NonLinPerturbations(background,
lp, zs, log10TAGN=...)``. MG params are injected via a mutable ``MGParams`` holder
bound by ``binned_mg_perturbations(...)``; the sampling wrapper updates mu/eta
before each ``loglike`` call. bin_index is fixed per run (as in the paper).

    mg = MGParams(mu=1.0, eta=1.0, bin_index=4)
    Lin, NonLin = binned_mg_perturbations(mg, MODEL_DIR,
                                          baseline_linear=LCDM.Linear,
                                          baseline_nonlinear=LCDM.NonLinear)
    # in the sampling wrapper, before each loglike:
    mg.mu, mg.eta = param_dict["mu"], param_dict["eta"]
"""

import os
import warnings
import numpy as np
from scipy import interpolate

_trapz = getattr(np, "trapezoid", np.trapz)  # numpy<2 compatibility

# Table 1 MG redshift bins: bin_index -> (zmin, zmax)
_BIN_EDGES = {
    0: (0.00, 0.43),
    1: (0.43, 0.91),
    2: (0.91, 1.47),
    3: (1.47, 2.15),
    4: (2.15, 3.00),
}

_EMU_CACHE = {}


def _load_emu(branch, bin_index, model_dir):
    """Load (and cache) a CosmoPower-JAX boost emulator. branch: 'linear'|'nonlinear'."""
    key = (branch, int(bin_index), os.path.abspath(model_dir))
    if key not in _EMU_CACHE:
        fp = os.path.join(model_dir, f"mg-boost-{branch}-bin{int(bin_index)}.npz")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"MG emulator not found: {fp}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

            # probe='custom_log' -> predict() returns 10**(NN output) = boost
            _EMU_CACHE[key] = CosmoPowerJAX(
                probe="custom_log", filepath=fp, verbose=False
            )
    return _EMU_CACHE[key]


def _boost_spline(emu, background, mu, eta, z):
    """Build a RectBivariateSpline B(z, k) from the emulator.

    The emulator k-grid is padded with constant edge values so the spline does
    *constant* (not divergent) extrapolation in k outside the trained range
    (the linear mu-boost is ~scale-independent there; high/low k are scale-cut).
    Input columns are stacked in the emulator's own parameter order, so the same
    helper handles the linear branch (has 'eta') and nonlinear branch (no 'eta').
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    src = {
        "Omega_m": background.Omega_cdm0 + background.Omega_b0,
        "Omega_b": background.Omega_b0,
        "h": background.H0 / 100.0,
        "ns": background.ns,
        "lnAs": np.log(background.As * 1e10),  # training convention
        "mu": float(mu),
        "eta": float(eta),
    }
    cols = [
        z if name == "z" else np.full(z.shape[0], src[name]) for name in emu.parameters
    ]
    boost = np.asarray(emu.predict(np.stack(cols, axis=1)))  # (nz, nk), de-logged
    k = np.asarray(emu.modes, dtype=float)
    k_pad = np.concatenate(([k[0] * 1e-3], k, [k[-1] * 1e3]))
    boost_pad = np.concatenate((boost[:, :1], boost, boost[:, -1:]), axis=1)
    return interpolate.RectBivariateSpline(z, k_pad, boost_pad, kx=1, ky=1)


def _sigma8(k, pk0):
    """sigma8 = sqrt[ 1/(2pi^2) int k^2 P(k,0) W^2(kR) dk ], R = 8 Mpc/h."""
    x = k * 8.0
    W = np.ones_like(x)
    m = x > 1e-8
    W[m] = 3.0 * (np.sin(x[m]) - x[m] * np.cos(x[m])) / x[m] ** 3
    return float(np.sqrt(_trapz(k**2 * pk0 * W**2, k) / (2.0 * np.pi**2)))


class MGParams:
    """Mutable holder for per-call MG params (mu, eta) and the fixed bin_index."""

    def __init__(self, mu=1.0, eta=1.0, bin_index=0):
        self.mu = float(mu)
        self.eta = float(eta)
        self.bin_index = int(bin_index)


def binned_mg_perturbations(mg_params, model_dir, baseline_linear, baseline_nonlinear):
    """Build cloelib-compatible (Linear, NonLinear) MG perturbation classes.

    mg_params : MGParams                 read at every instantiation (sampled mu/eta)
    model_dir : str                      dir with mg-boost-*-bin{0..4}.npz
    baseline_linear / baseline_nonlinear : LCDM perturbation classes, e.g.
        CosmoPowerJAXLCDMPerturbations.Linear / .NonLinear
    """

    class Linear:
        """Linear MG perturbations: boost_lin(k,z) * LCDM linear baseline."""

        def __init__(self, background, redshifts):
            assert background.Omega_k0 == 0, "Non-flat geometries not supported"
            self.background = background
            self.z = np.atleast_1d(np.asarray(redshifts, dtype=float))
            self.bin_index = mg_params.bin_index
            self.mu, self.eta = mg_params.mu, mg_params.eta
            self._base = baseline_linear(background, self.z)
            self._boost = _boost_spline(
                _load_emu("linear", self.bin_index, model_dir),
                background,
                self.mu,
                self.eta,
                self.z,
            )
            self.k = np.asarray(self._base.k)  # baseline (wide) grid

        def matter_power_spectrum(self, zs, ks):
            return self._boost(zs, ks) * np.asarray(
                self._base.matter_power_spectrum(zs, ks)
            )

        def growth_factor(self, zs, ks):
            return np.sqrt(
                self.matter_power_spectrum(zs, ks) / self.matter_power_spectrum(0.0, ks)
            )

        def sigma8_0(self):
            pk0 = self.matter_power_spectrum(0.0, self.k).flatten()
            return _sigma8(self.k, pk0)

    class NonLinear:
        """Nonlinear MG perturbations: boost_nl(k,z) * LCDM nonlinear baseline.

        Growth / sigma8 are derived from an internally-built *linear* MG P(k)
        (boost_lin * LCDM linear baseline), independent of the lp argument.
        """

        def __init__(self, background, linearperturbations, redshifts, log10TAGN=None):
            assert background.Omega_k0 == 0, "Non-flat geometries not supported"
            self.background = background
            self.z = np.atleast_1d(np.asarray(redshifts, dtype=float))
            self.bin_index = mg_params.bin_index
            self.mu, self.eta = mg_params.mu, mg_params.eta

            base_lin = baseline_linear(background, self.z)
            self._base_nl = baseline_nonlinear(
                background, base_lin, self.z, log10TAGN=log10TAGN
            )
            self._base_lin = base_lin
            self._boost_nl = _boost_spline(
                _load_emu("nonlinear", self.bin_index, model_dir),
                background,
                self.mu,
                self.eta,
                self.z,
            )
            self._boost_lin = _boost_spline(
                _load_emu("linear", self.bin_index, model_dir),
                background,
                self.mu,
                self.eta,
                self.z,
            )
            self.k = np.asarray(self._base_nl.k)  # wide grid -> full Limber support

        # --- nonlinear MG P(k): what enters the C(ell) ---
        def matter_power_spectrum(self, zs, ks):
            return self._boost_nl(zs, ks) * np.asarray(
                self._base_nl.matter_power_spectrum(zs, ks)
            )

        def matter_power_spectrum_cb(self, zs, ks):
            # mnu small in this analysis -> cb ~ total (approximate)
            return self.matter_power_spectrum(zs, ks)

        # --- linear MG P(k): used only for growth / sigma8 ---
        def _pk_lin(self, zs, ks):
            return self._boost_lin(zs, ks) * np.asarray(
                self._base_lin.matter_power_spectrum(zs, ks)
            )

        def growth_factor(self, zs, ks):
            return np.sqrt(self._pk_lin(zs, ks) / self._pk_lin(0.0, ks))

        def growth_rate(self):
            """Scale-independent f(z) = -(1+z) dlnD/dz from the linear MG P(k).

            Returns a 1-D array over self.z (matching CAMB/JAX backends); cloelib
            interpolates it onto the requested grid for the GCph RSD term.
            """
            k_ref = 0.05  # h/Mpc, linear & sub-horizon
            D = np.sqrt(
                self._pk_lin(self.z, k_ref).flatten() / float(self._pk_lin(0.0, k_ref))
            )
            return -(1.0 + self.z) * np.gradient(np.log(D), self.z)

        def sigma8_0(self):
            k = np.asarray(self._base_lin.k)
            return _sigma8(k, self._pk_lin(0.0, k).flatten())

        # --- modified lensing parameter (applied to WL kernel in photo.py) ---
        def _get_active_bin_mask(self, zs):
            zs = np.atleast_1d(zs)
            zmin, zmax = _BIN_EDGES[self.bin_index]
            return (zs > zmin) & (zs <= zmax)

        def Sigma(self, zs):
            r"""Sigma = mu(1+eta)/2 inside the active bin, 1 (LCDM) elsewhere."""
            zs = np.atleast_1d(zs)
            sigma = np.ones_like(zs, dtype=float)
            sigma[self._get_active_bin_mask(zs)] = self.mu * (1.0 + self.eta) / 2.0
            return sigma

    return Linear, NonLinear

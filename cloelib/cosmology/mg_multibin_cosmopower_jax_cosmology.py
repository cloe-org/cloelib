"""Multi-bin modified-gravity perturbations using CosmoPower-JAX boost emulators.

Like ``mg_cosmopower_jax_cosmology`` but for the *multi-bin* emulators, where mu
(and eta, linear only) vary in ALL 5 redshift bins simultaneously -- i.e. the
joint "unbinned" analysis. One emulator per sector (not per-bin), no bin_index.

    P_MG(k, z) = B(k, z; mu_1..5[, eta_1..5]) * P_LCDM(k, z)

applied as a multiplicative operator at query time on the baseline's k-grid, so
mu=eta=1 (GR) returns P_LCDM exactly.

Emulator inputs (stacked in the emulator's own parameter order):
    nonlinear:  [Omega_m, Omega_b, h, ns, lnAs, mu1..mu5, z]                (1024 k)
    linear:     [Omega_m, Omega_b, h, ns, lnAs, mu1..mu5, eta1..eta5, z]    (512 k)
  k in h/Mpc; lnAs = ln(1e10*As); probe='custom_log' -> predict() returns boost.
  eta has no nonlinear P(k) effect (not an NL input); it enters only via Sigma.

Sigma(z) is now a step function over the 5 bins:
    Sigma = mu_i (1 + eta_i) / 2   for z in bin i,   1 (GR) outside all bins.

Injecting mu/eta: cloelike's ``background`` carries no MG params, so a mutable
``MGParams`` holder (mu, eta as length-5 arrays) is bound by the factory; the
sampling wrapper sets mg.mu / mg.eta before each loglike call.

    mg = MGParams(mu=np.ones(5), eta=np.ones(5))
    Lin, NonLin = multibin_mg_perturbations(mg, MODEL_DIR,
                                            baseline_linear=LCDM.Linear,
                                            baseline_nonlinear=LCDM.NonLinear)
    # in the wrapper, before each loglike:
    mg.mu  = np.array([param_dict[f"mu{i}"]  for i in range(1, 6)])
    mg.eta = np.array([param_dict[f"eta{i}"] for i in range(1, 6)])
"""

import os
import warnings
import numpy as np
from scipy import interpolate

_trapz = getattr(np, "trapezoid", np.trapz)  # numpy<2 compatibility

# Table 1 MG redshift bins: index -> (zmin, zmax)
_BIN_EDGES = [(0.00, 0.43), (0.43, 0.91), (0.91, 1.47), (1.47, 2.15), (2.15, 3.00)]
N_BINS = 5

# Emulator filenames (adjust if you saved them under different names)
_EMU_FILES = {
    "linear": "mg-boost-linear-multibin.npz",
    "nonlinear": "mg-boost-nonlinear-multibin.npz",
}

_EMU_CACHE = {}


def _load_emu(branch, model_dir):
    """Load (and cache) a CosmoPower-JAX multi-bin boost emulator."""
    key = (branch, os.path.abspath(model_dir))
    if key not in _EMU_CACHE:
        fp = os.path.join(model_dir, _EMU_FILES[branch])
        if not os.path.exists(fp):
            raise FileNotFoundError(f"MG multi-bin emulator not found: {fp}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

            _EMU_CACHE[key] = CosmoPowerJAX(
                probe="custom_log", filepath=fp, verbose=False
            )
    return _EMU_CACHE[key]


def _boost_spline(emu, background, mu, eta, z):
    """RectBivariateSpline B(z, k) from the multi-bin emulator.

    Builds inputs in the emulator's own parameter order, so the NL branch
    (mu only) and the linear branch (mu + eta) are both handled automatically.
    k-grid padded with constant edge values for safe constant extrapolation.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    src = {
        "Omega_m": background.Omega_cdm0 + background.Omega_b0,
        "Omega_b": background.Omega_b0,
        "h": background.H0 / 100.0,
        "ns": background.ns,
        "lnAs": np.log(background.As * 1e10),
    }
    for i in range(N_BINS):
        src[f"mu{i + 1}"] = float(mu[i])
        src[f"eta{i + 1}"] = float(eta[i])
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
    """Mutable holder for the per-call MG parameters (length-5 mu and eta)."""

    def __init__(self, mu=None, eta=None):
        self.mu = np.ones(N_BINS) if mu is None else np.asarray(mu, dtype=float)
        self.eta = np.ones(N_BINS) if eta is None else np.asarray(eta, dtype=float)


def multibin_mg_perturbations(
    mg_params, model_dir, baseline_linear, baseline_nonlinear
):
    """Build cloelib-compatible (Linear, NonLinear) multi-bin MG perturbation classes.

    mg_params : MGParams                 read at every instantiation (sampled mu/eta)
    model_dir : str                      dir with the two multi-bin emulator .npz files
    baseline_linear / baseline_nonlinear : LCDM perturbation classes, e.g.
        CosmoPowerJAXLCDMPerturbations.Linear / .NonLinear
    """

    class Linear:
        """Linear MG perturbations: boost_lin(k,z; mu,eta) * LCDM linear baseline."""

        def __init__(self, background, redshifts):
            assert background.Omega_k0 == 0, "Non-flat geometries not supported"
            self.background = background
            self.z = np.atleast_1d(np.asarray(redshifts, dtype=float))
            self.mu, self.eta = mg_params.mu, mg_params.eta
            self._base = baseline_linear(background, self.z)
            self._boost = _boost_spline(
                _load_emu("linear", model_dir), background, self.mu, self.eta, self.z
            )
            self.k = np.asarray(self._base.k)

        def matter_power_spectrum(self, zs, ks):
            return self._boost(zs, ks) * np.asarray(
                self._base.matter_power_spectrum(zs, ks)
            )

        def growth_factor(self, zs, ks):
            return np.sqrt(
                self.matter_power_spectrum(zs, ks) / self.matter_power_spectrum(0.0, ks)
            )

        def sigma8_0(self):
            return _sigma8(self.k, self.matter_power_spectrum(0.0, self.k).flatten())

    class NonLinear:
        """Nonlinear MG perturbations: boost_nl(k,z; mu) * LCDM nonlinear baseline.

        Growth / sigma8 derived from an internally-built linear MG P(k).
        """

        def __init__(self, background, linearperturbations, redshifts, log10TAGN=None):
            assert background.Omega_k0 == 0, "Non-flat geometries not supported"
            self.background = background
            self.z = np.atleast_1d(np.asarray(redshifts, dtype=float))
            self.mu, self.eta = mg_params.mu, mg_params.eta

            base_lin = baseline_linear(background, self.z)
            self._base_nl = baseline_nonlinear(
                background, base_lin, self.z, log10TAGN=log10TAGN
            )
            self._base_lin = base_lin
            self._boost_nl = _boost_spline(
                _load_emu("nonlinear", model_dir), background, self.mu, self.eta, self.z
            )
            self._boost_lin = _boost_spline(
                _load_emu("linear", model_dir), background, self.mu, self.eta, self.z
            )
            self.k = np.asarray(self._base_nl.k)

        def matter_power_spectrum(self, zs, ks):
            return self._boost_nl(zs, ks) * np.asarray(
                self._base_nl.matter_power_spectrum(zs, ks)
            )

        def matter_power_spectrum_cb(self, zs, ks):
            return self.matter_power_spectrum(zs, ks)  # mnu small -> cb ~ total

        def _pk_lin(self, zs, ks):
            return self._boost_lin(zs, ks) * np.asarray(
                self._base_lin.matter_power_spectrum(zs, ks)
            )

        def growth_factor(self, zs, ks):
            return np.sqrt(self._pk_lin(zs, ks) / self._pk_lin(0.0, ks))

        def growth_rate(self):
            """1-D f(z) = -(1+z) dlnD/dz from the linear MG P(k) at k=0.05."""
            k_ref = 0.05
            D = np.sqrt(
                self._pk_lin(self.z, k_ref).flatten() / float(self._pk_lin(0.0, k_ref))
            )
            return -(1.0 + self.z) * np.gradient(np.log(D), self.z)

        def sigma8_0(self):
            k = np.asarray(self._base_lin.k)
            return _sigma8(k, self._pk_lin(0.0, k).flatten())

        def Sigma(self, zs):
            r"""Sigma(z) = mu_i(1+eta_i)/2 in bin i, 1 (GR) outside all bins."""
            zs = np.atleast_1d(zs)
            sigma = np.ones_like(zs, dtype=float)
            for i, (zmin, zmax) in enumerate(_BIN_EDGES):
                m = (zs > zmin) & (zs <= zmax)
                sigma[m] = self.mu[i] * (1.0 + self.eta[i]) / 2.0
            return sigma

    return Linear, NonLinear

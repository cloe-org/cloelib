"""Binned modified-gravity perturbations using CosmoPower-JAX boost emulators.

This module applies a modified-gravity *boost*

    B(k, z) = P_MG(k, z) / P_LCDM(k, z)

predicted by CosmoPower-JAX emulators on top of an external LCDM baseline
(another cloelib ``Perturbations`` object). The boost is applied as a
multiplicative operator *at query time* on the baseline's own k-grid:

    P_MG(k, z) = B(k, z) * P_LCDM(k, z)

so that at B = 1 (mu = eta = 1, GR) the result is the baseline LCDM spectrum
*exactly* -- the GR limit is recovered to machine precision, with no regridding
artefacts.

Two modes, selected by ``MGParams.bin_index``:

* **Single-bin** (``bin_index`` is an int): mu, eta vary in ONE redshift bin
  (the others held at GR). One emulator per bin::

      mg-boost-linear-bin{0..4}.npz     inputs [Omega_m,Omega_b,h,ns,lnAs,mu,eta,z]
      mg-boost-nonlinear-bin{0..4}.npz  inputs [Omega_m,Omega_b,h,ns,lnAs,mu,z]

* **Multi-bin** (``bin_index`` is ``None``): mu (and eta, linear only) vary in
  ALL 5 bins simultaneously -- the joint "unbinned" analysis. One emulator per
  sector::

      mg-boost-linear-multibin.npz     inputs [...,mu1..mu5, eta1..eta5, z]
      mg-boost-nonlinear-multibin.npz  inputs [...,mu1..mu5, z]

k is in h/Mpc; lnAs = ln(1e10 * As); emulators are loaded with
``probe='custom_log'`` so ``predict()`` returns the boost directly. eta has NO
nonlinear P(k) effect (it is not an NL input); it enters only via ``Sigma``.

Drop-in replacement for the ``LinPerturbations`` / ``NonLinPerturbations`` classes
in ``cloelike`` (``EuclidLikelihood_photo_Cls``). Provides the modified lensing
parameter ``Sigma(z) = mu(1 + eta)/2`` that ``photo.py`` applies to the WL kernel.

Injecting mu/eta
----------------
cloelike builds ``background`` from a fixed cosmo-key list (no mu/eta/bin_index)
and calls ``LinPerturbations(background, zs)`` / ``NonLinPerturbations(background,
lp, zs, log10TAGN=...)``. MG params are injected via a mutable ``MGParams`` holder
bound by ``mg_perturbations(...)``; the sampling wrapper updates mu/eta before
each ``loglike`` call. ``bin_index`` is fixed per run (as in the paper).

    # single-bin
    mg = MGParams(mu=1.0, eta=1.0, bin_index=4)
    Lin, NonLin = mg_perturbations(mg, MODEL_DIR,
                                   baseline_linear=LCDM.Linear,
                                   baseline_nonlinear=LCDM.NonLinear)
    # before each loglike:  mg.mu, mg.eta = param_dict["mu"], param_dict["eta"]

    # multi-bin
    mg = MGParams(mu=np.ones(5), eta=np.ones(5))          # bin_index=None
    Lin, NonLin = mg_perturbations(mg, MODEL_DIR, LCDM.Linear, LCDM.NonLinear)
    # before each loglike:
    #   mg.mu  = np.array([param_dict[f"mu{i}"]  for i in range(1, 6)])
    #   mg.eta = np.array([param_dict[f"eta{i}"] for i in range(1, 6)])

``binned_mg_perturbations`` (single-bin) and ``multibin_mg_perturbations``
(multi-bin) are kept as aliases of ``mg_perturbations`` for backwards
compatibility; the mode is chosen by ``mg_params.bin_index`` in all cases.
"""

import os
import warnings
import numpy as np
from scipy import interpolate

_trapz = getattr(np, "trapezoid", np.trapz)  # numpy<2 compatibility

# Table 1 MG redshift bins: index -> (zmin, zmax)
_BIN_EDGES = [(0.00, 0.43), (0.43, 0.91), (0.91, 1.47), (1.47, 2.15), (2.15, 3.00)]
N_BINS = len(_BIN_EDGES)

_EMU_CACHE = {}


def _emu_filename(branch, bin_index):
    """Emulator filename for a sector ('linear'|'nonlinear') and mode.

    ``bin_index=None`` -> the joint multi-bin emulator; an int -> that bin.
    """
    if bin_index is None:
        return f"mg-boost-{branch}-multibin.npz"
    return f"mg-boost-{branch}-bin{int(bin_index)}.npz"


def _load_emu(branch, model_dir, bin_index=None):
    """Load (and cache) a CosmoPower-JAX boost emulator.

    branch : 'linear' | 'nonlinear'
    bin_index : int for the per-bin (single-bin) emulator, or None for multi-bin.
    """
    key = (
        branch,
        bin_index if bin_index is None else int(bin_index),
        os.path.abspath(model_dir),
    )
    if key not in _EMU_CACHE:
        fp = os.path.join(model_dir, _emu_filename(branch, bin_index))
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
    """Build a RectBivariateSpline B(z, k) from a boost emulator.

    Handles both the single-bin emulators (scalar ``mu``/``eta`` -> parameter
    names ``'mu'``/``'eta'``) and the multi-bin emulators (length-N ``mu``/``eta``
    -> ``'mu1'..'muN'``/``'eta1'..'etaN'``). The right columns are selected from
    ``emu.parameters``, so the linear branch (uses eta) and nonlinear branch (no
    eta) are both handled automatically.

    The emulator k-grid is padded with constant edge values so the spline does
    *constant* (not divergent) extrapolation in k outside the trained range.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    eta = np.atleast_1d(np.asarray(eta, dtype=float))

    src = {
        "Omega_m": background.Omega_cdm0 + background.Omega_b0,
        "Omega_b": background.Omega_b0,
        "h": background.H0 / 100.0,
        "ns": background.ns,
        "lnAs": np.log(background.As * 1e10),  # training convention
    }
    if "mu" in emu.parameters:  # single-bin emulator
        src["mu"] = float(mu[0])
        src["eta"] = float(eta[0])
    else:  # multi-bin emulator: mu1..muN (and eta1..etaN for the linear branch)
        for i in range(mu.size):
            src[f"mu{i + 1}"] = float(mu[i])
        for i in range(eta.size):
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
    """Mutable holder for the per-call MG parameters.

    Single-bin:  ``MGParams(mu=1.0, eta=1.0, bin_index=i)``  -- scalar mu/eta in
                 bin ``i`` (the other bins held at GR).
    Multi-bin:   ``MGParams(mu=[...], eta=[...])``           -- length-``N_BINS``
                 arrays, ``bin_index=None`` (default). ``mu``/``eta`` default to
                 ``ones(N_BINS)`` (GR).
    """

    def __init__(self, mu=None, eta=None, bin_index=None):
        self.bin_index = None if bin_index is None else int(bin_index)
        if self.bin_index is None:
            self.mu = np.ones(N_BINS) if mu is None else np.asarray(mu, dtype=float)
            self.eta = np.ones(N_BINS) if eta is None else np.asarray(eta, dtype=float)
        else:
            self.mu = 1.0 if mu is None else float(mu)
            self.eta = 1.0 if eta is None else float(eta)


def _sigma_of_z(zs, mu, eta, bin_index):
    r"""Sigma(z) = mu(1+eta)/2 in the active bin(s), 1 (GR) elsewhere.

    Single-bin (``bin_index`` int): non-GR only inside that bin.
    Multi-bin (``bin_index`` None): step function over all bins.
    """
    zs = np.atleast_1d(np.asarray(zs, dtype=float))
    sigma = np.ones_like(zs, dtype=float)
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    if bin_index is None:  # multi-bin step function
        for i, (zmin, zmax) in enumerate(_BIN_EDGES):
            sigma[(zs > zmin) & (zs <= zmax)] = mu[i] * (1.0 + eta[i]) / 2.0
    else:  # single active bin
        zmin, zmax = _BIN_EDGES[bin_index]
        sigma[(zs > zmin) & (zs <= zmax)] = float(mu[0]) * (1.0 + float(eta[0])) / 2.0
    return sigma


def mg_perturbations(mg_params, model_dir, baseline_linear, baseline_nonlinear):
    """Build cloelib-compatible (Linear, NonLinear) MG perturbation classes.

    The single-bin vs multi-bin mode is chosen by ``mg_params.bin_index``
    (int -> single-bin, None -> multi-bin).

    mg_params : MGParams                 read at every instantiation (sampled mu/eta)
    model_dir : str                      directory holding the mg-boost-*.npz files
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
                _load_emu("linear", model_dir, self.bin_index),
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
                _load_emu("nonlinear", model_dir, self.bin_index),
                background,
                self.mu,
                self.eta,
                self.z,
            )
            self._boost_lin = _boost_spline(
                _load_emu("linear", model_dir, self.bin_index),
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
        def Sigma(self, zs):
            r"""Sigma = mu(1+eta)/2 in the active bin(s), 1 (LCDM) elsewhere."""
            return _sigma_of_z(zs, self.mu, self.eta, self.bin_index)

    return Linear, NonLinear


# Backwards-compatible aliases: mode is selected by mg_params.bin_index.
binned_mg_perturbations = mg_perturbations
multibin_mg_perturbations = mg_perturbations

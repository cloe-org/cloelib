"""Implementation of modified-gravity Perturbation cosmology using CosmoPower-JAX emulators.

The emulators predict the modified-gravity boost B(k, z) = P_MG(k, z) / P_LCDM(k, z),
applied on top of an external LCDM baseline (another cloelib Perturbations object)
as P_MG = B * P_LCDM on the baseline k-grid, so that mu = eta = 1 returns LCDM exactly.
mu and eta are read from a mutable MGParams holder and either modify a single redshift
bin (bin_index an int) or all bins at once (bin_index None); the matching emulators are
downloaded from Zenodo on first use. eta has no nonlinear effect and enters only through
the lensing parameter Sigma = mu(1 + eta)/2.
"""

# cloelib imports
from cloelib.cosmology.cosmology import growth_rate_on_redshifts

# General imports
import warnings
import numpy as np
from scipy import interpolate

_trapz = (
    np.trapezoid if hasattr(np, "trapezoid") else np.trapz
)  # numpy 2 removed np.trapz

# Zenodo record hosting the MG boost emulators, downloaded and cached on first use.
MG_EMULATOR_ZENODO_URL = "https://zenodo.org/records/22967046/files"

# Table 1 MG redshift bins: index -> (zmin, zmax)
_BIN_EDGES = [(0.00, 0.43), (0.43, 0.91), (0.91, 1.47), (1.47, 2.15), (2.15, 3.00)]
N_BINS = len(_BIN_EDGES)

_EMU_CACHE = {}

# Training-box ranges of the boost emulators; inputs outside the relevant box are
# rejected. The single-bin and multi-bin variants were trained over different ranges.
MG_EMULATOR_BOUNDS = {
    "single": {
        "Omega_m": (0.25, 0.35),
        "Omega_b": (0.040, 0.055),
        "h": (0.65, 0.73),
        "ns": (0.95, 1.00),
        "lnAs": (2.996, 3.091),
        "mu": (0.9, 1.1),
        "eta": (0.9, 1.1),
        "z": (0.01, 3.0),
    },
    "multi": {
        "Omega_m": (0.25, 0.40),
        "Omega_b": (0.040, 0.055),
        "h": (0.65, 0.75),
        "ns": (0.80, 1.20),
        "lnAs": (2.944, 3.219),
        "mu": (0.9, 1.1),
        "eta": (0.9, 1.1),
        "z": (0.0, 3.0),
    },
}


def _emu_filename(branch, bin_index):
    """Return the boost emulator filename for a sector and bin.

    Args:
        branch (str): Either 'linear' or 'nonlinear'.
        bin_index (Optional[int]): Bin index for the single-bin emulator, or None
            for the joint multi-bin emulator.

    Returns:
        str: The emulator file name.
    """
    if bin_index is None:
        return f"mg-boost-{branch}-multibin.npz"
    return f"mg-boost-{branch}-bin{int(bin_index)}.npz"


def _load_emu(branch, bin_index=None):
    """Load and cache a CosmoPower-JAX boost emulator.

    The emulator file is downloaded from the Zenodo record on first use and
    cached locally, mirroring ``cosmopower_jax_cosmology``.

    Args:
        branch (str): Either 'linear' or 'nonlinear'.
        bin_index (Optional[int]): Bin index for the single-bin emulator, or None
            for the multi-bin emulator.

    Returns:
        CosmoPowerJAX: The emulator, loaded with ``probe='custom_log'`` so that
            ``predict()`` returns the boost directly.
    """
    key = (branch, bin_index if bin_index is None else int(bin_index))
    if key not in _EMU_CACHE:
        from cloelib.cosmology.cosmopower_jax_cosmology import emulator_data

        fp = emulator_data(_emu_filename(branch, bin_index), MG_EMULATOR_ZENODO_URL)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

            _EMU_CACHE[key] = CosmoPowerJAX(
                probe="custom_log", filepath=fp, verbose=False
            )
    return _EMU_CACHE[key]


def _z_top(bin_index):
    """Return the highest redshift at which the MG modification is active.

    Above this redshift the boost is identically 1, and the emulator must not be
    queried there: the per-bin linear emulators are trained only up to the top of
    their own bin and extrapolate strongly beyond it, which would corrupt the
    growth factor used by the intrinsic-alignment kernel.

    Args:
        bin_index (Optional[int]): Active bin index, or None for the multi-bin case.

    Returns:
        float: The upper redshift edge of the active bin (or of the last bin).
    """
    if bin_index is None:
        return _BIN_EDGES[-1][1]
    return _BIN_EDGES[bin_index][1]


def _check_mg_bounds(emu, background, mu, eta, z):
    """Reject inputs outside the boost emulator's training box.

    The variant (single-bin or multi-bin) is detected from ``emu.parameters``,
    matching ``_boost_spline``. The lower z bound is not enforced, as z=0 is
    required for the growth and sigma8 normalisation.

    Args:
        emu (CosmoPowerJAX): The boost emulator.
        background (Background): Background cosmology.
        mu (array_like): Modified-gravity parameter(s).
        eta (array_like): Gravitational-slip parameter(s).
        z (array_like): Redshifts.

    Raises:
        ValueError: If any input lies outside the emulator's training range.
    """
    variant = "single" if "mu" in emu.parameters else "multi"
    box = MG_EMULATOR_BOUNDS[variant]
    cosmo = {
        "Omega_m": background.Omega_cdm0 + background.Omega_b0,
        "Omega_b": background.Omega_b0,
        "h": background.H0 / 100.0,
        "ns": background.ns,
        "lnAs": np.log(background.As * 1e10),
    }
    checks = list(cosmo.items()) + [("mu", mu), ("eta", eta), ("z", z)]
    for name, value in checks:
        low, high = box[name]
        values = np.atleast_1d(np.asarray(value, dtype=float))
        below = False if name == "z" else values.min() < low
        if below or values.max() > high:
            raise ValueError(
                f"MG emulator parameter {name} out of {variant}-bin "
                f"training range [{low}, {high}]."
            )


def _boost_spline(emu, background, mu, eta, z, z_top=None):
    """Build a bivariate spline B(z, k) from a boost emulator.

    Both the single-bin emulators (scalar ``mu``/``eta``) and the multi-bin
    emulators (length-N ``mu``/``eta``, named ``mu1..muN``/``eta1..etaN``) are
    handled; the columns are selected from ``emu.parameters``, so the linear
    branch (with eta) and nonlinear branch (without eta) both work. The emulator
    is evaluated only for ``z <= z_top``, with B = 1 above (see ``_z_top``);
    ``z_top=None`` disables the clamp.

    Args:
        emu (CosmoPowerJAX): The boost emulator.
        background (Background): Background cosmology.
        mu (array_like): Modified-gravity parameter(s).
        eta (array_like): Gravitational-slip parameter(s).
        z (array_like): Redshifts at which to build the spline.
        z_top (Optional[float]): Redshift above which B = 1, or None to disable.

    Returns:
        RectBivariateSpline: The boost B(z, k), with constant extrapolation in k.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    if z_top is not None:
        # Enforce the training box once, then query the emulator only where B is
        # nontrivial (z <= z_top) and fill B = 1 above, keeping the full z grid so
        # the spline knots stay strictly increasing.
        _check_mg_bounds(emu, background, mu, eta, z)
        active = z <= z_top
        k = np.asarray(emu.modes, dtype=float)
        k_pad = np.concatenate(([k[0] * 1e-3], k, [k[-1] * 1e3]))
        boost_pad = np.ones((z.size, k_pad.size))
        if active.any():
            z_q = z[active]
            if z_q.size == 1:  # a spline needs at least two z knots
                z_q = np.array([z_q[0], z_q[0] + 1e-3])
            spl_in = _boost_spline(emu, background, mu, eta, z_q, z_top=None)
            boost_pad[active] = spl_in(z[active], k_pad)
        return interpolate.RectBivariateSpline(z, k_pad, boost_pad, kx=1, ky=1)

    src = {
        "Omega_m": background.Omega_cdm0 + background.Omega_b0,
        "Omega_b": background.Omega_b0,
        "h": background.H0 / 100.0,
        "ns": background.ns,
        "lnAs": np.log(background.As * 1e10),
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
    boost = np.asarray(emu.predict(np.stack(cols, axis=1)))
    k = np.asarray(emu.modes, dtype=float)
    k_pad = np.concatenate(([k[0] * 1e-3], k, [k[-1] * 1e3]))
    boost_pad = np.concatenate((boost[:, :1], boost, boost[:, -1:]), axis=1)
    return interpolate.RectBivariateSpline(z, k_pad, boost_pad, kx=1, ky=1)


def _sigma8(k, pk0):
    """Compute sigma8 from a z=0 power spectrum with an R = 8 Mpc/h top hat.

    Args:
        k (numpy.ndarray): Wavenumbers in h Mpc^{-1}.
        pk0 (numpy.ndarray): Matter power spectrum at z=0.

    Returns:
        float: The rms matter fluctuation sigma8.
    """
    x = k * 8.0
    W = np.ones_like(x)
    m = x > 1e-8
    W[m] = 3.0 * (np.sin(x[m]) - x[m] * np.cos(x[m])) / x[m] ** 3
    return float(np.sqrt(_trapz(k**2 * pk0 * W**2, k) / (2.0 * np.pi**2)))


class MGParams:
    """Mutable holder for the per-call modified-gravity parameters mu and eta."""

    def __init__(self, mu=None, eta=None, bin_index=None):
        """Initialize the MGParams holder.

        Single-bin mode (``bin_index`` an int) stores scalar ``mu``/``eta`` for
        that bin; multi-bin mode (``bin_index=None``) stores length-``N_BINS``
        arrays. Unset ``mu``/``eta`` default to the GR value of one.
        """
        self.bin_index = None if bin_index is None else int(bin_index)
        if self.bin_index is None:
            self.mu = np.ones(N_BINS) if mu is None else np.asarray(mu, dtype=float)
            self.eta = np.ones(N_BINS) if eta is None else np.asarray(eta, dtype=float)
        else:
            self.mu = 1.0 if mu is None else float(mu)
            self.eta = 1.0 if eta is None else float(eta)


def _sigma_of_z(zs, mu, eta, bin_index):
    """Compute the lensing parameter Sigma(z) = mu(1 + eta)/2.

    Sigma is non-GR inside the active bin (single-bin) or a step function over all
    bins (multi-bin), and one elsewhere.

    Args:
        zs (array_like): Redshifts.
        mu (array_like): Modified-gravity parameter(s).
        eta (array_like): Gravitational-slip parameter(s).
        bin_index (Optional[int]): Active bin index, or None for the multi-bin case.

    Returns:
        numpy.ndarray: Sigma(z) on the input redshifts.
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


def mg_perturbations(mg_params, baseline_linear, baseline_nonlinear) -> tuple:
    """Build the modified-gravity Linear and NonLinear perturbation classes.

    The single-bin or multi-bin mode is selected by ``mg_params.bin_index``
    (an int or None). The returned classes follow the cloelike LinPerturbations
    and NonLinPerturbations protocols.

    Args:
        mg_params (MGParams): Holder read at each instantiation for the sampled
            mu and eta.
        baseline_linear (type): LCDM linear perturbation class, e.g.
            ``CosmoPowerJAXLCDMPerturbations.Linear``.
        baseline_nonlinear (type): LCDM nonlinear perturbation class, e.g.
            ``CosmoPowerJAXLCDMPerturbations.NonLinear``.

    Returns:
        tuple: The ``(Linear, NonLinear)`` modified-gravity perturbation classes.
    """

    class Linear:
        """Linear modified-gravity perturbations: linear boost times LCDM baseline."""

        def __init__(self, background, redshifts):
            """Initialize the Linear instance."""
            assert background.Omega_k0 == 0, "Non-flat geometries not supported"
            self.background = background
            self.z = np.atleast_1d(np.asarray(redshifts, dtype=float))
            self.bin_index = mg_params.bin_index
            self.mu, self.eta = mg_params.mu, mg_params.eta
            self._base = baseline_linear(background, self.z)
            self._boost = _boost_spline(
                _load_emu("linear", self.bin_index),
                background,
                self.mu,
                self.eta,
                self.z,
                z_top=_z_top(self.bin_index),
            )
            self.k = np.asarray(self._base.k)  # baseline (wide) grid

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the linear modified-gravity matter power spectrum.

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in h Mpc^{-1}.

            Returns:
                pk (numpy.ndarray): Linear matter power spectrum (boost times LCDM baseline).
            """
            return self._boost(zs, ks) * np.asarray(
                self._base.matter_power_spectrum(zs, ks)
            )

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the linear growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in h Mpc^{-1}.

            Returns:
                (numpy.ndarray): The growth factor.
            """
            return np.sqrt(
                self.matter_power_spectrum(zs, ks) / self.matter_power_spectrum(0.0, ks)
            )

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0 from the linear modified-gravity power spectrum.

            Returns:
                float: The rms matter fluctuation sigma8.
            """
            pk0 = self.matter_power_spectrum(0.0, self.k).flatten()
            return _sigma8(self.k, pk0)

    class NonLinear:
        """Nonlinear modified-gravity perturbations: nonlinear boost times LCDM baseline."""

        def __init__(self, background, linearperturbations, redshifts, log10TAGN=None):
            """Initialize the NonLinear instance."""
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
            z_top = _z_top(self.bin_index)
            self._boost_nl = _boost_spline(
                _load_emu("nonlinear", self.bin_index),
                background,
                self.mu,
                self.eta,
                self.z,
                z_top=z_top,
            )
            self._boost_lin = _boost_spline(
                _load_emu("linear", self.bin_index),
                background,
                self.mu,
                self.eta,
                self.z,
                z_top=z_top,
            )
            self.k = np.asarray(self._base_nl.k)  # wide grid -> full Limber support

        def matter_power_spectrum(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear modified-gravity matter power spectrum.

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in h Mpc^{-1}.

            Returns:
                pk (numpy.ndarray): Nonlinear matter power spectrum (boost times LCDM baseline).
            """
            return self._boost_nl(zs, ks) * np.asarray(
                self._base_nl.matter_power_spectrum(zs, ks)
            )

        def matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
            """Compute the nonlinear CDM+baryon matter power spectrum.

            Neutrino masses are small in this analysis, so the CDM+baryon spectrum
            is approximated by the total matter spectrum.

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in h Mpc^{-1}.

            Returns:
                pk (numpy.ndarray): Nonlinear CDM+baryon matter power spectrum.
            """
            return self.matter_power_spectrum(zs, ks)

        def _pk_lin(self, zs, ks):
            """Linear modified-gravity power spectrum, used for growth and sigma8.

            Built from the linear boost and the LCDM linear baseline, independent
            of the ``linearperturbations`` argument passed to the constructor.
            """
            return self._boost_lin(zs, ks) * np.asarray(
                self._base_lin.matter_power_spectrum(zs, ks)
            )

        def growth_factor(self, zs, ks) -> np.ndarray:
            """Compute the linear growth factor from the linear modified-gravity P(k).

            Args:
                zs (numpy.ndarray): Redshifts.
                ks (numpy.ndarray): Wavenumbers in h Mpc^{-1}.

            Returns:
                (numpy.ndarray): The growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].
            """
            return np.sqrt(self._pk_lin(zs, ks) / self._pk_lin(0.0, ks))

        def growth_rate(self, zs=None, ks=None) -> np.ndarray:
            """Compute the scale-independent growth rate f(z) = -(1 + z) dlnD/dz.

            f is computed on ``self.z`` from the linear modified-gravity power
            spectrum, then interpolated to ``zs`` and broadcast over ``ks``.

            Args:
                zs (Optional[numpy.ndarray]): Redshifts. Defaults to the instance grid.
                ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

            Returns:
                (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
                    and (nz, nk) otherwise.
            """
            k_ref = 0.05  # h/Mpc, linear & sub-horizon
            pk_lin_ref_0 = np.ravel(self._pk_lin(0.0, k_ref))[0]
            D = np.sqrt(self._pk_lin(self.z, k_ref).flatten() / pk_lin_ref_0)
            f = -(1.0 + self.z) * np.gradient(np.log(D), self.z)
            return growth_rate_on_redshifts(self.z, f, zs, ks)

        def sigma8_0(self) -> float:
            """Compute sigma8 at z=0 from the linear modified-gravity power spectrum.

            Returns:
                float: The rms matter fluctuation sigma8.
            """
            k = np.asarray(self._base_lin.k)
            return _sigma8(k, self._pk_lin(0.0, k).flatten())

        def Sigma(self, zs) -> np.ndarray:
            """Compute the modified lensing parameter Sigma(z) = mu(1 + eta)/2.

            Args:
                zs (numpy.ndarray): Redshifts.

            Returns:
                (numpy.ndarray): Sigma(z), non-GR in the active bin(s) and one elsewhere.
            """
            return _sigma_of_z(zs, self.mu, self.eta, self.bin_index)

    return Linear, NonLinear

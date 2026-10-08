"""CosmoPower-JAX emulators for the linear and nonlinear matter power spectrum.

Uses ``cosmopower_jax`` (rather than the TensorFlow-based ``cosmopower``) for fast
JAX-accelerated predictions. Covers LCDM, wCDM and w0waCDM, plus curvature and
running-spectral-index extensions, each over neutrino configurations N_mnu = 0-3.
Every cosmology exposes ``Linear`` and ``NonLinear`` inner classes; the baseline
models additionally offer halofit nonlinear variants. Each provides both the
total-matter spectrum (``matter_power_spectrum``) and the CDM+baryon spectrum
(``matter_power_spectrum_cb``, built at construction from the paired cb emulator),
along with ``sigma8``, ``fsigma8``, ``growth_factor``, ``growth_factor_cb`` and
``growth_rate``, and
validates inputs against ``CP_EMULATOR_BOUNDS``.
"""

from cloelib.cosmology.cosmology import (
    Background,
    Perturbations,
    growth_rate_on_redshifts,
)
from cloelib.auxiliary.extrapolator import extend_spectra
from cloelib.auxiliary.math_utils import ensure_z_zero_included

import numpy as np
from scipy import interpolate
import os
import urllib.request
import warnings
from typing import Optional


# GitHub repository hosting the emulator files, downloaded and cached on first use.
# This is the raw-file base path; each .npz is fetched individually, not the whole repo.
GITHUB_EMULATOR_URL = (
    "https://raw.githubusercontent.com/cosmopower-organization/"
    "Euclid-DR1-matter-emulators/main/emulators"
)


def _emulator_url_for(filename: str) -> str:
    """Return the GitHub raw base URL, including subfolder, that hosts ``filename``.

    Emulators are organised by cosmology family and nonlinear prescription:
    ``<model>/hmcode``, ``<model>/halofit`` and ``extended/<variant>``. The
    subfolder is derived from the filename prefix.
    """
    if filename.startswith("halofit-"):
        subdir = f"{filename.split('-')[1]}/halofit"
    elif filename.startswith("nrun-"):
        subdir = "extended/running"
    elif filename.startswith("curvature-"):
        subdir = "extended/curvature"
    else:
        subdir = f"{filename.split('-')[0]}/hmcode"
    return f"{GITHUB_EMULATOR_URL}/{subdir}"


def emulator_data(filename: str, base_url: Optional[str] = None) -> str:
    """Download the emulator data file if it does not exist.

    Parameters
    ----------
    filename : str
        The name of the file to download.
    base_url : str, optional
        The base URL from which to download the file. Defaults to the GitHub
        location derived from the filename.

    Returns
    -------
    str
        The path to the downloaded file.
    """
    if base_url is None:
        base_url = _emulator_url_for(filename)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "emulator-data-jax")
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        url = f"{base_url}/{filename}"
        print(f"Downloading {filename} from {url} ...")
        urllib.request.urlretrieve(url, file_path)

    return file_path


def load_pk_emulator(filepath: str):
    """Load a CosmoPower-JAX P(k) emulator from an .npz file.

    Uses probe='custom_log' since P(k) emulators predict log10(P(k)).

    Parameters
    ----------
    filepath : str
        Full path to the .npz emulator file.

    Returns
    -------
    CosmoPowerJAX
        The loaded emulator object.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

        return CosmoPowerJAX(probe="custom_log", filepath=filepath, verbose=False)


def load_sigma_emulator(filepath: str):
    """Load a CosmoPower-JAX sigma8/fsigma8 emulator from an .npz file.

    Uses probe='custom' since sigma8 emulators predict values directly (not log).

    Parameters
    ----------
    filepath : str
        Full path to the .npz emulator file.

    Returns
    -------
    CosmoPowerJAX
        The loaded emulator object.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        from cosmopower_jax.cosmopower_jax import CosmoPowerJAX

        return CosmoPowerJAX(probe="custom", filepath=filepath, verbose=False)


CP_EMULATOR_BOUNDS = {
    "ombh2": np.array([0.001, 0.1]),
    "omch2": np.array([0.05, 0.9]),
    "H0": np.array([20.0, 100.0]),
    "ns": np.array([0.6, 1.3]),
    "lnAs": np.array([1.61, 5.0]),
    "mnu": np.array([0.0, 1.0]),
    "logT_AGN": np.array([7.3, 8.5]),
    "w0": np.array([-3.0, 1.0]),
    "wa": np.array([-3.0, 3.0]),
    "w": np.array([-3.0, 0.0]),
    "omk": np.array([-0.3, 0.3]),
    "alpha_s": np.array([-0.3, 0.3]),
}


def check_emulator_bounds(params: dict) -> None:
    """Validate emulator input parameters against their training-box bounds.

    Parameters
    ----------
    params : dict
        Mapping of parameter name to its scalar value. Keys not present in
        ``CP_EMULATOR_BOUNDS`` (e.g. ``z``) are skipped.

    Raises
    ------
    ValueError
        If any parameter lies outside its emulator training range.
    """
    for key, value in params.items():
        if key not in CP_EMULATOR_BOUNDS:
            continue
        if np.prod(np.asarray(value) - CP_EMULATOR_BOUNDS[key]) > 0:
            raise ValueError(f"Parameter {key} out of emulator range.")


def _extended_pk_cb(cb_file, params, z_emu, extrap_z, ns):
    """Predict and extend the cb P(k, z) grid from a CosmoPower-JAX cb emulator.

    Built with the same ``extend_spectra`` options as the total-matter spectrum,
    so the returned grid lands on the same ``(z, k)`` grid as ``Pk``.
    """
    emu = load_pk_emulator(emulator_data(cb_file))
    Pk = np.array(emu.predict(params))
    _, _, Pk_out = extend_spectra(
        np.asarray(emu.modes),
        z_emu,
        Pk,
        flag_range=True,
        option_wavenumber="logk2",
        option_redshift="power_law",
        extrap_z=extrap_z,
        option_cosmo="const",
        ns=ns,
    )
    return Pk_out


# --------------------------------------------------------------------------- #
# Shared implementation of the emulator classes
#
# Every cosmology below exposes the same ``Linear`` / ``NonLinear`` (and, for the
# baseline models, ``NonLinearHalofit``) classes. Their behaviour is identical up
# to (i) which emulator files are loaded and (ii) which extra cosmological
# parameters the emulator takes, so each method is implemented once here and
# installed on the (otherwise empty) inner classes by ``_emulator_class``.
# --------------------------------------------------------------------------- #

_SECTORS = {
    # sector: (P(k) file, cb P(k) file, sigma8/fsigma8 file, takes log10TAGN)
    "linear": (
        "{p}-{n}mass-linear.npz",
        "{p}-{n}mass-cb-linear.npz",
        "{p}-{n}mass-s8-fs8.npz",
        False,
    ),
    "nonlinear": (
        "{p}-{n}mass-nonlinear.npz",
        "{p}-{n}mass-cb-nonlinear.npz",
        "{p}-{n}mass-s8-fs8.npz",
        True,
    ),
    "halofit": (
        "halofit-{p}-{n}mass-nonlinear.npz",
        "halofit-{p}-{n}mass-cb-nonlinear.npz",
        "halofit-{p}-{n}mass-s8-fs8.npz",
        False,
    ),
}

_NEUTRINOS = {
    0: "no massive neutrinos",
    1: "one massive neutrino with mnu={mnu}eV",
    2: "two degenerate massive neutrinos with mnu={mnu}eV",
    3: "three degenerate massive neutrinos with mnu={mnu}eV",
}


class _ModelSpec:
    """What distinguishes one emulated cosmology from another.

    Args:
        label (str): Human-readable cosmology name used in ``__str__``.
        prefix (str): Emulator file prefix, e.g. ``"lcdm"`` or ``"curvature-w0wa"``.
        n_mnu (tuple): Supported numbers of massive neutrino species.
        params (dict): Extra emulator inputs beyond the base LCDM set, mapping the
            emulator parameter name to the ``Background`` attribute it is read from.
        flat_only (bool): Whether the emulator only supports Omega_k0 = 0.
    """

    def __init__(self, label, prefix, n_mnu, params, flat_only):
        """Store the model description."""
        self.label = label
        self.prefix = prefix
        self.n_mnu = tuple(n_mnu)
        self.params = dict(params)
        self.flat_only = flat_only


def _emulator_files(spec, sector, n_mnu):
    """Return the (P(k), cb P(k), sigma8) emulator files for one configuration.

    Args:
        spec (_ModelSpec): The cosmology.
        sector (str): One of ``"linear"``, ``"nonlinear"`` or ``"halofit"``.
        n_mnu (int): Number of massive neutrino species.

    Returns:
        tuple: The three file names.

    Raises:
        ValueError: If ``n_mnu`` is not supported by this cosmology.
    """
    if n_mnu not in spec.n_mnu:
        supported = ", ".join(str(n) for n in spec.n_mnu)
        raise ValueError(f"Unsupported N_mnu={n_mnu}. Supported: {supported}")
    pk, cb, sigma, _ = _SECTORS[sector]
    return tuple(f.format(p=spec.prefix, n=n_mnu) for f in (pk, cb, sigma))


def _setup(self, background, redshifts, log10TAGN=None):
    """Load the emulators and build the P(k), P_cb(k) and sigma8 tables.

    Shared by the linear, nonlinear and halofit ``__init__`` methods; the sector
    and cosmology are read from the class attributes set by ``_emulator_class``.
    """
    spec, sector = self._spec, self._sector
    cp_file, cp_file_cb, cp_file_sigma = _emulator_files(spec, sector, background.N_mnu)
    cp_file, cp_file_cb, cp_file_sigma = (
        emulator_data(cp_file),
        emulator_data(cp_file_cb),
        emulator_data(cp_file_sigma),
    )
    self.has_neutrinos = background.N_mnu > 0

    emulator = load_pk_emulator(cp_file)
    if sector == "linear":
        self.cp_LIN = emulator
    else:
        self.cp_NONLIN = emulator
    self.cp_SIGMA = load_sigma_emulator(cp_file_sigma)
    self.k_emu = np.asarray(emulator.modes)
    self.k_min = self.k_emu[0]
    self.k_max = self.k_emu[-1]
    self.background = background
    if spec.flat_only:
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

    self.z = ensure_z_zero_included(redshifts[redshifts <= 5])
    self.params = {
        "ombh2": self.background.Omega_b0 * self.background.h**2,
        "omch2": self.background.Omega_cdm0 * self.background.h**2,
        "H0": self.background.H0,
        "ns": self.background.ns,
        "lnAs": np.log(self.background.As * 1e10),
    }
    for name, attribute in spec.params.items():
        self.params[name] = getattr(self.background, attribute)
    baryons = _SECTORS[sector][3]
    if baryons:
        self.params["logT_AGN"] = log10TAGN if log10TAGN is not None else 7.6
    if self.has_neutrinos:
        self.params["mnu"] = self.background.mnu

    check_emulator_bounds(self.params)

    for key in self.params.keys():
        self.params[key] = np.tile(self.params[key], len(self.z))
    self.params["z"] = self.z

    if baryons:
        sigma_params = self.params
    else:
        # The sigma8/fsigma8 emulator is shared with the HMcode modules and expects
        # logT_AGN; sigma8/fsigma8 are (near) linear quantities, so they are
        # evaluated at the fiducial logT_AGN=7.6.
        self.sigma_params = self.params.copy()
        self.sigma_params["logT_AGN"] = np.tile(7.6, len(self.z))
        sigma_params = self.sigma_params

    Pk = np.array(emulator.predict(self.params))
    k_out, z_out, Pk_out = extend_spectra(
        self.k_emu,
        self.z,
        Pk,
        flag_range=True,
        option_wavenumber="logk2",
        option_redshift="power_law",
        extrap_z=redshifts,
        option_cosmo="const",
        ns=self.background.ns,
    )
    self.k, self.z, self.Pk = k_out, z_out, Pk_out
    self.Pk_interp = interpolate.RectBivariateSpline(self.z, self.k, Pk_out, kx=1, ky=1)
    self.Pk_cb = _extended_pk_cb(
        cp_file_cb, self.params, self.params["z"], redshifts, self.background.ns
    )
    self.Pk_cb_interp = interpolate.RectBivariateSpline(
        self.z, self.k, self.Pk_cb, kx=1, ky=1
    )

    sigma_predictions = np.array(self.cp_SIGMA.predict(sigma_params))
    self.sigma8 = sigma_predictions[:, 0]
    self.fsigma8 = sigma_predictions[:, 1]


def _init_linear(self, background: Background, redshifts: np.ndarray):
    """Initialize the Linear instance."""
    _setup(self, background, redshifts)


def _init_nonlinear(
    self,
    background: Background,
    linearperturbations: Perturbations,
    redshifts: np.ndarray,
    log10TAGN: Optional[float] = None,
):
    """Initialize the NonLinear instance.

    ``log10TAGN`` controls the HMcode2020 baryonic feedback; the halofit
    prescription is dark-matter-only, so there it is accepted for interface
    compatibility but ignored.
    """
    # Retained for downstream consumers that need the *linear* P(k) back from a
    # tracer's (nonlinear) `perturbations` - same attribute/pattern as
    # HMcode2020Emu, EE2, BACCOemu, Emantis and JAX.
    self.linearperturbations = linearperturbations
    _setup(self, background, redshifts, log10TAGN)


def _str(self) -> str:
    """Return emulator description."""
    sector = {
        "linear": "linear",
        "nonlinear": "nonlinear",
        "halofit": "nonlinear (halofit)",
    }
    n_mnu = self.background.N_mnu
    neutrinos = _NEUTRINOS.get(n_mnu, "unsupported").format(mnu=self.background.mnu)
    extra = "".join(
        f", {name}={getattr(self.background, attribute)}"
        for name, attribute in self._spec.params.items()
    )
    text = (
        f"Cosmopower-JAX {sector[self._sector]} P(k) module for {self._spec.label} "
        f"cosmology.\nConfiguration: N_mnu={n_mnu}, {neutrinos}{extra}."
    )
    if self._sector == "halofit":
        text += " Dark-matter-only halofit (no baryonic feedback)."
    return text


def _matter_power_spectrum(self, zs, ks) -> np.ndarray:
    """Compute the matter power spectrum P(k, z).

    Args:
        zs (numpy.ndarray): Redshifts.
        ks (numpy.ndarray): Wavenumbers in Mpc^-1.

    Returns:
        pk (numpy.ndarray): Total-matter power spectrum.
    """
    return self.Pk_interp(zs, ks)


def _matter_power_spectrum_cb(self, zs, ks) -> np.ndarray:
    """Compute the CDM+baryon matter power spectrum P_cb(k, z).

    Args:
        zs (numpy.ndarray): Redshifts.
        ks (numpy.ndarray): Wavenumbers in Mpc^-1.

    Returns:
        pk (numpy.ndarray): CDM+baryon (no neutrino) power spectrum.
    """
    return self.Pk_cb_interp(zs, ks)


def _growth_factor(self, zs, ks) -> np.ndarray:
    """Compute the growth factor D(z, k) = sqrt[P(z, k) / P(0, k)].

    Args:
        zs (numpy.ndarray): Redshifts.
        ks (numpy.ndarray): Wavenumbers in Mpc^-1.

    Returns:
        (numpy.ndarray): The growth factor, normalised to D(z=0) = 1.
    """
    return np.sqrt(self.Pk_interp(zs, ks) / self.Pk_interp(0, ks))


def _growth_factor_cb(self, zs, ks) -> np.ndarray:
    """Compute the cb growth factor D_cb(z, k) = sqrt[P_cb(z, k) / P_cb(0, k)].

    Args:
        zs (numpy.ndarray): Redshifts.
        ks (numpy.ndarray): Wavenumbers in Mpc^-1.

    Returns:
        (numpy.ndarray): The cb growth factor, normalised to D_cb(z=0) = 1.
    """
    return np.sqrt(self.Pk_cb_interp(zs, ks) / self.Pk_cb_interp(0, ks))


def _growth_rate(
    self, zs: Optional[np.ndarray] = None, ks: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the scale-independent growth rate f(z) = fsigma8 / sigma8.

    Args:
        zs (Optional[numpy.ndarray]): Redshifts, interpolated on the grid the
            emulator was evaluated at. Defaults to that grid.
        ks (Optional[numpy.ndarray]): Wavenumbers used to broadcast f.

    Returns:
        (numpy.ndarray): The growth rate, with shape (nz,) if ks is None
            and (nz, nk) otherwise.
    """
    return growth_rate_on_redshifts(
        self.params["z"], self.fsigma8 / self.sigma8, zs, ks
    )


def _sigma8_0(self) -> float:
    """Compute sigma8 at z=0.

    Returns:
        float: The rms matter fluctuation sigma8 at z=0.
    """
    return self.sigma8[0]


def _emulator_class(spec, sector):
    """Return a class decorator installing the shared emulator implementation.

    The decorated (empty) class gains ``__init__``, ``__str__`` and the
    ``Perturbations`` methods, all shared between every cosmology and sector.

    Args:
        spec (_ModelSpec): The cosmology.
        sector (str): One of ``"linear"``, ``"nonlinear"`` or ``"halofit"``.

    Returns:
        Callable: The class decorator.
    """

    def decorate(cls):
        cls._spec = spec
        cls._sector = sector
        cls.__init__ = _init_linear if sector == "linear" else _init_nonlinear
        cls.__str__ = _str
        cls.matter_power_spectrum = _matter_power_spectrum
        cls.matter_power_spectrum_cb = _matter_power_spectrum_cb
        cls.growth_factor = _growth_factor
        cls.growth_factor_cb = _growth_factor_cb
        cls.growth_rate = _growth_rate
        cls.sigma8_0 = _sigma8_0
        return cls

    return decorate


_BASE = (0, 1, 2, 3)
_EXTENDED = (0, 1, 3)
_W0WA = {"w0": "w0", "wa": "wa"}

_LCDM = _ModelSpec("LCDM", "lcdm", _BASE, {}, flat_only=True)
_WCDM = _ModelSpec("wCDM", "wcdm", _BASE, {"w": "w0"}, flat_only=True)
_W0WACDM = _ModelSpec("w0waCDM", "w0wa", _BASE, _W0WA, flat_only=True)
_LCDM_CURVATURE = _ModelSpec(
    "LCDM+curvature", "curvature-lcdm", _EXTENDED, {"omk": "Omega_k0"}, flat_only=False
)
_W0WA_CURVATURE = _ModelSpec(
    "w0waCDM+curvature",
    "curvature-w0wa",
    _EXTENDED,
    {**_W0WA, "omk": "Omega_k0"},
    flat_only=False,
)
_LCDM_RUNNING = _ModelSpec(
    "LCDM+running spectral index",
    "nrun-lcdm",
    _EXTENDED,
    {"alpha_s": "alpha_s"},
    flat_only=False,
)
_W0WA_RUNNING = _ModelSpec(
    "w0waCDM+running spectral index",
    "nrun-w0wa",
    _EXTENDED,
    {**_W0WA, "alpha_s": "alpha_s"},
    flat_only=False,
)


class CosmoPowerJAXw0waCDMPerturbations:
    """
    Class for w0waCDM cosmology perturbations using CosmoPower-JAX emulators.
    """

    @_emulator_class(_W0WACDM, "linear")
    class Linear:
        """Linear matter power spectrum."""

    @_emulator_class(_W0WACDM, "nonlinear")
    class NonLinear:
        """Nonlinear matter power spectrum using HMcode2020."""

    @_emulator_class(_W0WACDM, "halofit")
    class NonLinearHalofit:
        """Nonlinear matter power spectrum using halofit (Takahashi 2012)."""


class CosmoPowerJAXwCDMPerturbations:
    """
    Class for wCDM cosmology perturbations using CosmoPower-JAX emulators.
    """

    @_emulator_class(_WCDM, "linear")
    class Linear:
        """Linear matter power spectrum."""

    @_emulator_class(_WCDM, "nonlinear")
    class NonLinear:
        """Nonlinear matter power spectrum using HMcode2020."""

    @_emulator_class(_WCDM, "halofit")
    class NonLinearHalofit:
        """Nonlinear matter power spectrum using halofit (Takahashi 2012)."""


class CosmoPowerJAXLCDMPerturbations:
    """Class for LCDM cosmology perturbations using CosmoPower-JAX emulators."""

    @_emulator_class(_LCDM, "linear")
    class Linear:
        """Linear matter power spectrum."""

    @_emulator_class(_LCDM, "nonlinear")
    class NonLinear:
        """Nonlinear matter power spectrum using HMcode2020."""

    @_emulator_class(_LCDM, "halofit")
    class NonLinearHalofit:
        """Nonlinear matter power spectrum using halofit (Takahashi 2012)."""


class CosmoPowerJAXLCDMCurvaturePerturbations:
    """Class for LCDM+curvature cosmology perturbations using CosmoPower-JAX.

    Emulators cover spatial curvature Omega_k0.
    Neutrino configuration is selected by ``background.N_mnu``: 0 (massless), 1 (one massive), 3 (three degenerate).
    The k-mode grid is read from each emulator (``.modes``); inputs are
    validated against :data:`CP_EMULATOR_BOUNDS`.
    """

    @_emulator_class(_LCDM_CURVATURE, "linear")
    class Linear:
        """Linear matter power spectrum."""

    @_emulator_class(_LCDM_CURVATURE, "nonlinear")
    class NonLinear:
        """Nonlinear matter power spectrum using HMcode2020."""


class CosmoPowerJAXw0waCurvaturePerturbations:
    """Class for w0waCDM+curvature cosmology perturbations using CosmoPower-JAX.

    Emulators cover spatial curvature Omega_k0 together with a w0waCDM dark energy background (free w0, wa).
    Neutrino configuration is selected by ``background.N_mnu``: 0 (massless), 1 (one massive), 3 (three degenerate).
    The k-mode grid is read from each emulator (``.modes``); inputs are
    validated against :data:`CP_EMULATOR_BOUNDS`.
    """

    @_emulator_class(_W0WA_CURVATURE, "linear")
    class Linear:
        """Linear matter power spectrum."""

    @_emulator_class(_W0WA_CURVATURE, "nonlinear")
    class NonLinear:
        """Nonlinear matter power spectrum using HMcode2020."""


class CosmoPowerJAXLCDMRunningIndexPerturbations:
    """Class for LCDM+running spectral index cosmology perturbations using CosmoPower-JAX.

    Emulators cover running spectral index alpha_s.
    Neutrino configuration is selected by ``background.N_mnu``: 0 (massless), 1 (one massive), 3 (three degenerate).
    The k-mode grid is read from each emulator (``.modes``); inputs are
    validated against :data:`CP_EMULATOR_BOUNDS`.
    """

    @_emulator_class(_LCDM_RUNNING, "linear")
    class Linear:
        """Linear matter power spectrum."""

    @_emulator_class(_LCDM_RUNNING, "nonlinear")
    class NonLinear:
        """Nonlinear matter power spectrum using HMcode2020."""


class CosmoPowerJAXw0waRunningIndexPerturbations:
    """Class for w0waCDM+running spectral index cosmology perturbations using CosmoPower-JAX.

    Emulators cover running spectral index alpha_s together with a w0waCDM dark energy background (free w0, wa).
    Neutrino configuration is selected by ``background.N_mnu``: 0 (massless), 1 (one massive), 3 (three degenerate).
    The k-mode grid is read from each emulator (``.modes``); inputs are
    validated against :data:`CP_EMULATOR_BOUNDS`.
    """

    @_emulator_class(_W0WA_RUNNING, "linear")
    class Linear:
        """Linear matter power spectrum."""

    @_emulator_class(_W0WA_RUNNING, "nonlinear")
    class NonLinear:
        """Nonlinear matter power spectrum using HMcode2020."""

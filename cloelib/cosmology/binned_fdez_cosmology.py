"""Implementation of Background and Perturbation cosmology for binned density of dark energy."""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.auxiliary.extrapolator import extend_spectra
from cloelib.cosmology.cosmology import Perturbations

# General imports
import numpy as np
from typing import Optional
from scipy import interpolate
import jax.numpy as jnp

from cloelib.cosmology.theta_h0 import equivalent_wcdm_growth
from cloelib.cosmology.theta_h0 import theta_to_H0 as _theta_to_H0

# Gauss–Legendre nodes for the comoving-distance integral. Inside one f_DE bin
# the integrand is analytic, and panels no wider than `_CHI_PANEL` stay at the
# roundoff of scipy.quad (which is what this method used to call per step).
_CHI_GL_X, _CHI_GL_W = np.polynomial.legendre.leggauss(16)
_CHI_PANEL = 2.0

# Cosmology imports
try:
    import cloelib.auxiliary.jaxmgrowth_de_descrete as mgrowth
except ImportError:
    raise ImportError("MGrowth could not be imported or initialised.")

try:
    import HMcode2020Emu as hmcodeemu

    HM2020_emu = hmcodeemu.Matter_powerspectrum()
    redshift_max = HM2020_emu.emulator["linear"]["bounds"]["z"][1]
except ImportError:
    raise ImportError("HMcode2020emu could not be imported or initialised.")


def get_z_bin(z, zbin_edges):
    """
    Returns the bin index for a given redshift value.

    Parameters:
    -----------
    z : float or array-like
        Redshift value(s) to bin
    zbin_edges : array-like
        Array of bin edges (must be sorted in ascending order)

    Returns:
    --------
    bin_index : int or array
        Bin index (0-indexed). Returns -1 for values below the first edge
        and len(zbin_edges)-2 for values at or above the last edge.
    """
    # Check if input is scalar
    is_scalar = np.isscalar(z)

    # Convert to array for processing
    z = np.asarray(z)
    zbin_edges = np.asarray(zbin_edges)

    # np.digitize returns the index of the bin (1-indexed)
    # right=False means left edge inclusive, right edge exclusive
    bin_idx = np.digitize(z, zbin_edges, right=False) - 1

    # Handle edge cases:
    # - Values below first edge get -1
    # - Values at or above last edge get the last bin index
    bin_idx = np.clip(bin_idx, -1, len(zbin_edges) - 2)

    # For values exactly at the last edge, put them in the last bin
    if is_scalar:
        if z >= zbin_edges[-1]:
            bin_idx = len(zbin_edges) - 2
    else:
        bin_idx[z >= zbin_edges[-1]] = len(zbin_edges) - 2

    # Return scalar if input was scalar, otherwise return array
    return bin_idx.item() if is_scalar else bin_idx


def _first_bin_fde(z, z1, f_hi, w0_transition=-1.0):
    """Cubic from 1 at ``z = 0`` to ``f_hi`` at ``z1``, flat at ``z1``.

    ``w0_transition`` is ``w(z=0)`` of this ramp, not the cosmological
    ``w_0``. It sets the slope through ``df/dz(0) = 3(w0_transition+1)``.
    The default ``-1`` has zero slope at the origin and reproduces the
    smoothstep ``x^2 (3-2x)``. Past ``z1`` the value stays at ``f_hi`` and
    the slope is zero, so the cubic meets the constant part of the bin.
    """
    z_arr = np.asarray(z, dtype=float)
    B = 3.0 * (float(w0_transition) + 1.0)
    C = 3.0 * (f_hi - 1.0) / z1**2 - 2.0 * B / z1
    D = -2.0 * (f_hi - 1.0) / z1**3 + B / z1**2
    z_use = np.minimum(z_arr, z1)
    fde = 1.0 + B * z_use + C * z_use**2 + D * z_use**3
    dfdz = B + 2.0 * C * z_use + 3.0 * D * z_use**2
    dfdz = np.where(z_arr >= z1, 0.0, dfdz)
    fde = np.where(z_arr >= z1, f_hi, fde)
    return fde, dfdz


def get_de_density(z, zbin_edges, fde_i, fde_transition=0.0, w0_transition=-1.0):
    """Binned ``f_DE(z)``, anchored to 1 at ``z = 0``.

    ``E^2(z) = Omega_m0 (1+z)^3 + (1 - Omega_m0) f_DE(z)`` equals 1 at
    ``z = 0`` only if ``f_DE(0) = 1``. Up to the centre of the first bin,
    ``f_DE`` is a cubic fixed by that value, by ``w(0) = w0_transition``, and by
    reaching ``fde_i[0]`` with zero slope. ``w0_transition=-1`` is the smoothstep.
    The rest of the first bin stays at ``fde_i[0]``.

    ``fde_transition = 0`` leaves every later bin at its constant ``fde_i``.
    The growth integrator then rescales ``dD/da`` by ``(H_before/H_after)^2``
    at each of those edges. ``fde_transition`` in ``(0, 1]`` replaces each
    edge by the same smoothstep, of width ``fde_transition`` times the
    narrower adjacent bin. 
    """
    transition = float(fde_transition)
    if transition < 0.0 or transition > 1.0:
        raise ValueError(
            f"fde_transition must be between 0 and 1, got {transition}"
        )
    z_arr = np.asarray(z, dtype=float)
    scalar = z_arr.ndim == 0
    z_arr = np.atleast_1d(z_arr)
    fde = np.asarray(fde_i, dtype=float)
    edges = np.asarray(zbin_edges, dtype=float)
    z1 = float(0.5 * (edges[1] + edges[0])) #float(edges[1])
    if z1 <= 0.0:
        raise ValueError(
            "the first f_DE bin must end at z > 0 so f_DE can transition to 1 at z=0"
        )

    bins = np.maximum(get_z_bin(z_arr, edges), 0)
    rho = fde[bins]
    ramp, _ = _first_bin_fde(z_arr, z1, fde[0], w0_transition)
    rho = np.where(z_arr < z1, ramp, rho)
    if transition > 0.0:
        for k in range(1, len(fde)):
            half = 0.5 * transition * min(
                edges[k] - edges[k - 1], edges[k + 1] - edges[k]
            )
            if half <= 0.0:
                continue
            zedge = edges[k]
            inside = (z_arr >= zedge - half) & (z_arr < zedge + half)
            x = (z_arr - (zedge - half)) / (2.0 * half)
            s = x * x * (3.0 - 2.0 * x)
            # The cubic ends at the bin centre, so every edge, including the
            # first, joins two plateaus.
            f_left = fde[k - 1]
            blended = (1.0 - s) * f_left + s * fde[k]
            rho = np.where(inside, blended, rho)
    if scalar:
        return rho[0]
    return rho


def get_de_density_i(z, zbin_edges, fde_i, fde_transition=0.0, w0_transition=-1.0):
    return get_de_density(z, zbin_edges, fde_i, fde_transition, w0_transition)

def theta_to_H0(
    theta,
    omch2,
    ombh2,
    fde_i,
    zbin_edges,
    omega_k=0.0,
    mnu=0.06,
    TCMB=2.7255,
    nnu=3.044,
    h0_bracket=(50.0, 90.0),
    fde_transition=0.0,
    w0_transition=-1.0,
):
    """H0 in km/s/Mpc for this binned ``f_DE(z)``, given ``theta = r_s / D_M(z_*)``."""
    return _theta_to_H0(
        theta,
        omch2,
        ombh2,
        lambda z: get_de_density(z, zbin_edges, fde_i, fde_transition, w0_transition),
        omega_k=omega_k,
        mnu=mnu,
        TCMB=TCMB,
        nnu=nnu,
        h0_bracket=h0_bracket,
    )
class DEBinnedDensityBackground:
    """Beyond w0wa-background cosmological calculations."""

    def __init__(
        self,
        cosmology_dict: dict,
        zbin_edges: np.ndarray, 
        fde_i: np.ndarray,
        fde_transition: float = 0.0,
        w0_transition: float = -1.0,
    ) -> None:
        """
        Initialize the CAMBBackground instance with cosmological parameters.

        Args:
            H0 (float): Hubble parameter in [km/s/Mpc].
            Omega_b0 (float): Baryonic matter density parameter.
            Omega_cdm0 (float): Cold dark matter density parameter.
            Omega_k0(float): Curvature density parameter.
            As (float): Scalar amplitude of primordial fluctuations.
            ns (float): Scalar spectral index.
            mnu (float): Total sum of neutrino mass in [eV].
            zbin_edges (np.ndarray): Array of bin edges for the redshift bins.
            fde_i (np.ndarray): ``f_DE`` in each redshift bin. The first entry
                is reached at the centre of the first bin; below that,
                ``f_DE`` is a cubic from 1 at ``z = 0``.
            fde_transition (float): ``0`` keeps the later bins as steps.
                A value in ``(0, 1]`` smooths each later edge over that
                fraction of the narrower adjacent bin.
            w0_transition (float): ``w(z=0)`` of the first-bin cubic, not the
                cosmological ``w_0``. The default ``-1`` makes that cubic flat
                at ``z = 0``.
            binning (str): Type of binning used for the redshift bins.
        """
        self.zbin_edges = np.asarray(zbin_edges, dtype=float)
        if self.zbin_edges.size < 2 or self.zbin_edges[1] <= 0.0:
            raise ValueError(
                "zbin_edges must contain a first bin ending at z > 0, "
                "so f_DE can transition to 1 at z=0."
            )
        self.zbin_widths = np.diff(self.zbin_edges)
        self.zbin_centers = 0.5 * (self.zbin_edges[1:] + self.zbin_edges[:-1])
        self.fde_i = np.asarray(fde_i, dtype=float)
        self.fde_transition = float(fde_transition)
        if self.fde_transition < 0.0 or self.fde_transition > 1.0:
            raise ValueError(
                f"fde_transition must be between 0 and 1, got {self.fde_transition}"
            )
        self.w0_transition = float(w0_transition)

        self.H0 = cosmology_dict["H0"]
        self.h = self.H0 / 100.0
        self.Omega_b0 = cosmology_dict["Omega_b0"]
        self.Omega_cdm0 = cosmology_dict["Omega_cdm0"]
        self.Omega_k0 = cosmology_dict["Omega_k0"]
        self.mnu = cosmology_dict["mnu"]
        self.As = cosmology_dict["As"]
        self.ns = cosmology_dict["ns"]
        self.Omega_m0 = self.Omega_b0 + self.Omega_cdm0 + self.mnu / 93.14 / self.h**2
        self.N_mnu = cosmology_dict["N_mnu"]
        self.N_eff = cosmology_dict["N_eff"]

    def hubble_parameter(self, zs: np.ndarray, units: str = "km/s/Mpc") -> np.ndarray:
        """
        Return the Hubble parameter as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.
            units (str): Units for the Hubble parameter ('1/Mpc' or 'km/s/Mpc').

        Returns:
            np.ndarray: Hubble parameter values at specified redshifts.
        """
        DE_z_grid = self._fde(zs)
        # f_DE(0) = 1, so this radicand is 1 at z = 0.
        E_z_grid = np.sqrt(
            self.Omega_m0 * pow(1.0 + zs, 3) + (1 - self.Omega_m0) * DE_z_grid
        )
        if units == "1/Mpc":
            return E_z_grid * self.H0 / SPEED_OF_LIGHT * 1000
        if units == "km/s/Mpc":
            return E_z_grid * self.H0
        raise ValueError(
            "Unsupported units for hubble_parameter. Choose '1/Mpc' or 'km/s/Mpc'."
        )

    def _expansion(self, zs: np.ndarray) -> np.ndarray:
        """E(z) = H(z)/H0, with the same f_DE(z) as ``hubble_parameter``."""
        fde = self._fde(zs)
        return np.sqrt(self.Omega_m0 * (1.0 + zs) ** 3 + (1.0 - self.Omega_m0) * fde)

    def comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the comoving distance as a function of redshift.

        The integral is the same one previously evaluated with ``scipy.quad``
        on each redshift step. It is now a 16-point Gauss–Legendre rule on
        panels split at the f_DE bin edges (and capped at width 2), which
        matches that quadrature to its own tolerance. Repeated calls with the
        same redshifts reuse the result.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Comoving distance values.
        """
        zs = np.asarray(zs, dtype=float)
        key = (zs.shape, np.ascontiguousarray(zs).tobytes())
        cached = getattr(self, "_comoving_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1].copy()

        z_max = float(np.max(zs)) if zs.size else 0.0
        breaks = self.zbin_edges[(self.zbin_edges > 0.0) & (self.zbin_edges < z_max)]
        if self.fde_transition > 0.0:
            edges = self.zbin_edges
            windows = []
            for k in range(1, len(edges) - 1):
                half = 0.5 * self.fde_transition * min(
                    edges[k] - edges[k - 1], edges[k + 1] - edges[k]
                )
                windows.extend((edges[k] - half, edges[k] + half))
            breaks = np.concatenate((np.ravel(breaks), windows))
            breaks = breaks[(breaks > 0.0) & (breaks < z_max)]
        nodes = np.unique(np.concatenate((np.array([0.0]), np.ravel(zs), breaks)))
        pieces = [nodes[0]]
        for left, right in zip(nodes[:-1], nodes[1:]):
            n_panel = int(np.ceil((right - left) / _CHI_PANEL))
            pieces.extend(np.linspace(left, right, n_panel + 1)[1:])
        nodes = np.asarray(pieces)
        left, right = nodes[:-1], nodes[1:]
        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        z = mid[:, None] + half[:, None] * _CHI_GL_X
        inv_e = 1.0 / self._expansion(z)
        chi_nodes = np.cumsum(half * (inv_e @ _CHI_GL_W)) * (
            SPEED_OF_LIGHT / 1000.0
        ) / self.H0
        index = np.searchsorted(nodes, zs)
        chi = np.zeros(zs.shape, dtype=float)
        positive = index > 0
        chi[positive] = chi_nodes[index[positive] - 1]
        self._comoving_cache = (key, chi)
        return chi.copy()

    def transverse_comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the transverse comoving distance between two redshifts.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Transverse comoving distance values.
        """
        x = self.comoving_distance(zs)

        if self.Omega_k0 == 0.0:
            y = x
            return y
        else:
            raise ValueError(
                "Omega_k0 != 0.0 not supported for DEBackground.transverse_comoving_distance."
            )

    def angular_diameter_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the angular diameter distance as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Angular diameter distance values.
        """
        return self.transverse_comoving_distance(zs) / (1 + zs)

    def Omega_m(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the matter density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Matter density values.
        """
        DE_z_grid = self._fde(zs)
        E2_z_grid = self.Omega_m0 * pow(1.0 + zs, 3) + (1 - self.Omega_m0) * DE_z_grid
        return self.Omega_m0 * pow(1.0 + zs, 3) / E2_z_grid

    def Omega_b(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the baryon density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Baryonic density values at specified redshifts.
        """
        DE_z_grid = self._fde(zs)
        E2_z_grid = self.Omega_m0 * pow(1.0 + zs, 3) + (1 - self.Omega_m0) * DE_z_grid
        return self.Omega_b0 * pow(1.0 + zs, 3) / E2_z_grid

    @property
    def rdrag(self) -> float:
        """Sound horizon radius at last scattering in Mpc."""
        w_nu = 0.0107
        Obh2 = self.Omega_b0 * self.h**2
        Och2 = self.Omega_cdm0 * self.h**2
        rdrag = (
            55.154
            * np.exp(-72.3 * (w_nu * self.mnu + 0.0006) ** 2)
            / (Obh2**0.12807 * (Obh2 + Och2) ** 0.25351)
        )
        return rdrag

    def _fde(self, zs):
        return get_de_density(
            zs, self.zbin_edges, self.fde_i, self.fde_transition, self.w0_transition
        )


class DEBinnedDensityLinearPerturbations:
    """Class for linear perturbations using MGrowth, inheriting from Perturbations parent class."""

    def __init__(
        self, background: DEBinnedDensityBackground, base_linear_perturbations: Perturbations
    ) -> None:
        assert background.Omega_k0 == 0, "Non flat geometries not supported"
        # must be binned-DE
        self.background = background
        # must be LCDM
        assert (
            base_linear_perturbations.background.w0 == -1.0
            and base_linear_perturbations.background.wa == 0.0
        ), "Base linear perturbations must be LCDM"
        self.base = base_linear_perturbations

        # is called later in kernels, has to match
        self.z = self.base.z

        # Sort scale factors and redshifts in ascending order (early to late times)
        # hardcoded
        self.z_sorted = np.linspace(0.0, 5.0, 256, endpoint=True)
        # Get scale factors
        self.a = 1.0 / (1.0 + self.z_sorted)
        self.a_sorted = self.a[::-1]

        # hardcoded to save time for f(R) by setting 128 k-values and interpolating later
        self.k = np.logspace(
            np.log10(self.base.k[0]), np.log10(self.base.k[-1]), 128, endpoint=True
        )  # in 1/Mpc
        self.k_len = len(self.k)

        self.zbin_edges = background.zbin_edges
        self.zbin_widths = background.zbin_widths
        self.zbin_centers = background.zbin_centers
        self.fde_i = background.fde_i
        self.a_grid = jnp.linspace(1e-4, 1.0, 512)
        z_grid = 1.0 / np.linspace(1e-4, 1.0, 512) - 1.0
        self.fde_transition = background.fde_transition
        self.w0_transition = background.w0_transition
        self.fde_vals = jnp.asarray(
            get_de_density(
                z_grid,
                background.zbin_edges,
                self.fde_i,
                self.fde_transition,
                self.w0_transition,
            )
        )

        # Build background dict for MGrowth
        background_mgrowth = {
            "Omega_m": self.background.Omega_m(np.array([0.0]))[0],
            "h": self.background.h,
            "w0": -1.0,
            "wa": 0.0,
            "a_arr": self.a_sorted,
        }
        # Compute MG and LCDM growth and assign linear growth parameters
        self._compute_growth(background_mgrowth)

    def _compute_growth(self, bg_dict):
        """Handle model-specific MGrowth and LCDM growth evaluation."""
        cosmo = mgrowth.fde_a(bg_dict)
        # growth expects per-bin amplitudes (len = n_bins), not the a-grid samples
        D_raw, f_raw = cosmo.growth_parameters(
            jnp.asarray(self.fde_i),
            jnp.asarray(self.zbin_edges),
            self.fde_transition,
            self.w0_transition,
        )
        D_mg, f_mg = (
            np.repeat(D_raw[::-1, None], self.k_len, axis=1),
            np.repeat(f_raw[::-1, None], self.k_len, axis=1),
        )


        # Get LCDM growth and construct array over k to be applied as normalisation
        base_cosmo = mgrowth.LCDM(bg_dict)
        D_base_raw, _ = base_cosmo.growth_parameters()
        D_base = D_base_raw[::-1]

        # Interpolate D(z, k) and f(z, k)
        self.dz_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, D_mg, kx=1, ky=1
        )
        self.dz_norm_dz0_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, D_mg / D_mg[0, :], kx=1, ky=1
        )
        self.fz_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, f_mg, kx=1, ky=1
        )
        self.dz_norm_lcdm_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, D_mg / D_base[0, None], kx=1, ky=1
        )
        self.dz_norm_lcdm_z_interp = interpolate.RectBivariateSpline(
            self.z_sorted, self.k, D_mg / D_base[:, None], kx=1, ky=1
        )

    def growth_factor(self, zs, ks) -> np.ndarray:
        """Calculate the growth factor D(z, k) normalized to D(0).

        Parameters
        ----------
        zs : array_like
            Redshifts at which to calculate the growth factor.
        ks : array_like
            Wavenumbers at which to calculate the growth factor.

        Returns
        -------
        np.ndarray
            The growth factor as a function of redshift and wavenumber.
        """
        return self.dz_norm_dz0_interp(zs, ks)

    def growth_rate(self, zs, ks) -> np.ndarray:
        """Calculate the growth rate f(z, k) for given redshifts and wavenumbers.

        Parameters
        ----------
        zs : array_like
            Redshifts at which to calculate the growth rate.
        ks : array_like
            Wavenumbers at which to calculate the growth rate.

        Returns
        -------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """
        return self.fz_interp(zs, ks)

    def matter_power_spectrum(self, zs, ks) -> np.ndarray:
        """Compute the linear matter power spectrum.

        Parameters
        ----------
        ks : numpy.ndarray
            Wavenumber in h Mpc^{-1}.
        zs : numpy.ndarray
            Redshifts.

        Returns
        -------
        np.ndarray
            Linear matter power spectrum at the specified redshifts and scales.
        """
        ps_base = self.base.matter_power_spectrum(0.0, ks)
        Dz_div_Dlcdmz0 = self.dz_norm_lcdm_interp(zs, ks)
        return Dz_div_Dlcdmz0**2 * ps_base

    def sigma8_0(self) -> float:
        """
        Calculate the sigma8 value for the current cosmology.

        Returns:
        --------
        float
            The sigma8 value.
        """
        Dz_div_Dlcdmz0 = self.dz_norm_lcdm_interp(0.0, 0.01)
        return self.base.sigma8_0() * Dz_div_Dlcdmz0


class DEBinnedDensityNonlinearPerturbations:
    """Class for nonlinear (pseudo) perturbations using MGrowth, inheriting from Perturbations parent class."""

    def __init__(
        self,
        background: DEBinnedDensityBackground,
        linearperturbations: Perturbations,
        redshifts: np.ndarray,
        log10TAGN: Optional[float] = None,
        w_eq: Optional[bool] = True,
    ):
        """Initialize the HMemuNonLinearPerturbations intance."""
        assert background.Omega_k0 == 0, "Non flat geometries not supported"

        redshift_max = HM2020_emu.emulator["nonlinear"]["bounds"]["z"][1]

        self.z = redshifts[redshifts <= redshift_max]
        # binned DE background
        self.background = background
        # binned DE linear perturbations
        self.de_lin = linearperturbations

        if w_eq:
            w0, dz_wcdm = equivalent_wcdm_growth(
                self.background,
                self.z,
                lambda z: get_de_density(
                    z,
                    self.background.zbin_edges,
                    self.background.fde_i,
                    self.background.fde_transition,
                    self.background.w0_transition,
                ),
            )
            # D of this model over D of the matched constant-w model.
            # dz_interp is k-independent; 0.01 only picks one column.
            d_model = self.de_lin.dz_interp(self.z, np.full(len(self.z), 0.01))[:, 0]
            rescale_factor = d_model / dz_wcdm
        else:
            rescale_factor = self.de_lin.dz_norm_lcdm_z_interp(
                self.z, np.full(len(self.z), 0.01)
            )[:, 0]
            w0 = np.tile(-1.0, len(self.z))

        self.params_hm_emu = {
            "omega_cdm": self.background.Omega_cdm0,
            "omega_baryon": self.background.Omega_b0,
            # modify for pseudo-power spectrum
            "As": self.background.As * rescale_factor ** 2,
            "ns": self.background.ns,
            "hubble": self.background.H0 / 100,
            "neutrino_mass": self.background.mnu,
            "w0": w0,
            "wa": 0.0,
        }
        baryonic_boost = log10TAGN is not None

        if baryonic_boost:
            self.params_hm_emu["log10TAGN"] = log10TAGN

        hm_bounds = HM2020_emu.emulator["nonlinear"]["bounds"]

        for key in self.params_hm_emu.keys():
            if key not in ["As", "w0"]:
                if np.prod(self.params_hm_emu[key] - hm_bounds[key]) > 0:
                    raise ValueError("HMcode 2020 NL emulator out of range.")
                else:
                    self.params_hm_emu[key] = np.tile(
                        self.params_hm_emu[key], len(self.z)
                    )
            else:
                check = np.array(
                    [
                        np.prod(As_i - hm_bounds["As"])
                        for As_i in self.params_hm_emu["As"]
                    ]
                )
                if (check > 0).any():
                    raise ValueError("As re-scaled is out of range.")
                check = np.array(
                    [
                        np.prod(w0_i - hm_bounds["w0"])
                        for w0_i in self.params_hm_emu["w0"]
                    ]
                )
                if (check > 0).any():
                    raise ValueError("w0 equality for re-scaling is out of range.")

        self.params_hm_emu["z"] = self.z

        _, Pk = HM2020_emu.get_nonlinear_pk(
            nonu=False, **self.params_hm_emu, baryonic_boost=baryonic_boost
        )

        k_emu = HM2020_emu.emulator["nonlinear"]["k"] * self.background.h

        # Low-k extrapolation.
        # Done this way to use Pk array instead of calling an interpolator
        # This only works if the redshift array is exactly the same within
        # range. This should be, but we should probably make sure in some way
        Pk_lin_mask_k = linearperturbations.k < k_emu[0]
        Pk_lin_mask_z = linearperturbations.z <= redshift_max
        Pk_lin = linearperturbations.matter_power_spectrum(
            linearperturbations.z, linearperturbations.k
        )[Pk_lin_mask_z][:, Pk_lin_mask_k]
        k_all = np.concatenate((linearperturbations.k[Pk_lin_mask_k], k_emu))
        Pk_all = np.concatenate((Pk_lin, self.background.h**-3 * Pk), axis=1)

        # Warning: a lot of parameters currently hard-coded
        k_out, z_out, Pk_out = extend_spectra(
            k_all,
            self.z,
            Pk_all,
            flag_range=True,
            option_wavenumber="power_law",
            option_redshift="power_law",
            extrap_z=redshifts,
            option_cosmo="const",
            ns=self.background.ns,
        )

        self.k = k_out
        self.z = z_out
        self.Pk = Pk_out

        pk_interp = interpolate.RectBivariateSpline(self.z, self.k, self.Pk, kx=1, ky=1)

        self.Pk_interp = pk_interp

    def matter_power_spectrum(self, zs, ks):
        r"""Compute the linear matter power spectrum.

        Parameters
        ----------
        ks: numpy.ndarray
            Wave number in h Mpc^{-1}

        zs: numpy.ndarray
            redshifts

        Returns
        -------
        pk: numpy.ndarray
            Linear matter power spectrum at the specified scale
            and redshift

        """
        return self.Pk_interp(zs, ks)

    def growth_factor(self, zs, ks) -> np.ndarray:
        r"""Calculate the growth factor for given redshifts and wavenumbers.

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
        return self.de_lin.growth_factor(zs, ks)

    def growth_rate(self, zs, ks) -> np.ndarray:
        """
        Calculate the growth rate for given redshifts and wavenumbers.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """

        return self.de_lin.growth_rate(zs, ks)

    def sigma8_0(self) -> float:
        """Retrieve sigma8 at z=0."""

        return self.de_lin.sigma8_0()

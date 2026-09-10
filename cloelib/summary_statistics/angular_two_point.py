"""Module for angular two-point functions."""

# cloelib imports
from cloelib.observables.photo.tracer import Tracer
from cloelib.observables.photo import PositionsTracer
from cloelib.observables.photo import ShearTracer
from cloelib.observables.cmb import CMBLensingTracer
from cloelib.observables.photo.spectrum_engine import (
    build_spectra_bank,
    get_effective_pk,
    needs_generalized_engine,
)
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.auxiliary.math_utils import simpsons_weights_jit
from cloelib.profiling import profile_function

# General imports
import interpax
import jax.numpy as np
import jax

# results imports
from cosmolib.data import AngularPowerSpectrum, COSEBI


@jax.jit
def Cl_integration(WT1, WT2, Pkl, H, chi2, weights) -> jax.numpy.ndarray:
    """
    Perform the integration to compute the angular power spectrum Cl.

    The integration is done using the unnormalized trapezoidal rule,
    utilizing the window functions, power spectrum, Hubble parameter,
    and comoving distance squared.

    Parameters:
        WT1 (jax.numpy.ndarray): Window function for the first tracer.
        WT2 (jax.numpy.ndarray): Window function for the second tracer.
        Pkl (jax.numpy.ndarray): Matter power spectrum interpolated on Limber grid.
        H (jax.numpy.ndarray): Hubble parameter evaluated at redshifts.
        chi2 (jax.numpy.ndarray): Square of comoving distances at redshifts.
        weights (jax.numpy.ndarray): Array of weights used for the fixed nodes integration.

    Returns:
        (jax.numpy.ndarray): Angular power spectrum Cl with shape (len(ells), len(ells), len(ells)).
    """
    return np.einsum("iz,jz,lz,z,z,z->lij", WT1, WT2, Pkl, 1 / H, 1 / chi2, weights)


@jax.jit
def Pkl_interp(k_l, z_l, ks, zs, Pk) -> jax.numpy.ndarray:
    """
    Interpolate the matter power spectrum on a Limber grid.

    Utilizes interpax's 2D interpolation with Akima method to handle
    non-uniform grids in logarithmic space. Extrapolation is enabled
    for values outside the given grid.

    Parameters:
        k_l (jax.numpy.ndarray): Wavenumbers corresponding to (ells + 0.5) / chi.
        z_l (jax.numpy.ndarray): Redshift grid for Limber integration.
        ks (jax.numpy.ndarray): Original wavenumber grid of the matter power spectrum.
        zs (jax.numpy.ndarray): Original redshift grid of the matter power spectrum.
        Pk (jax.numpy.ndarray): Matter power spectrum values on (ks, zs) grid.

    Returns:
        (jax.numpy.ndarray): Interpolated power spectrum on the Limber grid.
    """
    return 10 ** interpax.interp2d(
        jax.numpy.log10(k_l),
        z_l,
        jax.numpy.log10(ks),
        zs,
        jax.numpy.log10(Pk),
        method="akima",
        extrap=True,
    )


Pkl_interp_vmap = jax.jit(jax.vmap(Pkl_interp, in_axes=(0, None, None, None, None)))


@jax.jit
def Pkl_interp_signed(k_l, z_l, ks, zs, Pk) -> jax.numpy.ndarray:
    """
    Interpolate a possibly-negative effective power spectrum on a Limber grid.

    `Pkl_interp` interpolates `log10(P)` in log-log `(log10(k), z)` space,
    which assumes `P > 0` everywhere - true for the plain matter power
    spectrum, but not for e.g. a matter-intrinsic (`GI`) effective spectrum,
    which is generically signed (`C1(z)` in the TATT/NLA model carries a
    minus sign - see `tatt.py`).

    A first attempt at this interpolated `P` itself (not `log10(P)`)
    linearly to sidestep `log10` of a negative number; that turned out to
    be a real bug, not just a simplification - `P(k,z)` spans many orders
    of magnitude across a log-spaced `k` grid, and interpax's akima
    interpolation of the raw values across that range was wildly
    inaccurate (verified: it produced results independent of the actual
    effective-Pk terms, off by 5-6 orders of magnitude from the correct
    C1**2 * P_dd limit). This keeps log-log accuracy for the *magnitude*
    (`log10(|P|)`) and interpolates the sign separately, recombining
    `sign(interpolated sign) * 10**(interpolated log10|P|)` - standard
    practice for interpolating a signed, log-scale quantity.

    Used by the generalized engine (`AngularTwoPoint._compute_cl_
    generalized`) instead of `Pkl_interp`, which stays log-log (and
    untouched) for the always-positive legacy path.

    Parameters/Returns: as `Pkl_interp`.
    """
    log_abs_pk = jax.numpy.log10(jax.numpy.abs(Pk) + 1e-300)
    sign_pk = jax.numpy.sign(Pk)

    log_ks = jax.numpy.log10(ks)
    # A real (non-power-law) effective spectrum - e.g. TATT's one-loop
    # kernels, which are steep and sign-changing near the edges of their
    # k-grid, unlike the smooth matter Pk `Pkl_interp` extrapolates - has no
    # well-defined asymptotic shape past its own grid to extrapolate at all:
    # akima extrapolation there can send the *extrapolated* log-magnitude to
    # +-hundreds at extreme Limber wavenumbers (k_l = (ell+0.5)/chi blows up
    # as chi -> 0, i.e. the z ~ 0 edge of the redshift grid, at high ell),
    # which `10**(...)` either overflows to +-inf (poisoning the whole Cl
    # sum) or, if merely clipped after the fact, leaves an astronomically
    # large-but-finite value that still dominates the sum - found via a
    # genuine bad `Cl` from a real FAST-PT-backed `TATTContribution`
    # (`PBJ_tatt.py`), not the smoother placeholder kernels, which never
    # extrapolate steeply enough to trigger this. Clamping the query k to
    # the grid's own domain before interpolating - constant (edge-value)
    # behavior past the grid, standard practice for a function with no
    # known extrapolation law - avoids extrapolating this kind of kernel at
    # all; the clamped region only ever affects the physically negligible
    # chi~0 edge (both the `dz` weight and the `1/chi**2` Limber prefactor
    # already suppress it there), not genuine bulk grid values.
    log_k = jax.numpy.clip(jax.numpy.log10(k_l), log_ks.min(), log_ks.max())

    interp_log_abs = interpax.interp2d(
        log_k, z_l, log_ks, zs, log_abs_pk, method="akima", extrap=True
    )
    interp_sign = interpax.interp2d(
        log_k, z_l, log_ks, zs, sign_pk, method="akima", extrap=True
    )
    return jax.numpy.sign(interp_sign) * 10**interp_log_abs


Pkl_interp_signed_vmap = jax.jit(
    jax.vmap(Pkl_interp_signed, in_axes=(0, None, None, None, None))
)


@jax.jit
def Cl_int_liz_jz(WT1l, WT2, Pkl, invH, invchi2, weights):
    # for window w/ RSD X window w/o RSD
    return np.einsum("liz,jz,lz,z,z,z->lij", WT1l, WT2, Pkl, invH, invchi2, weights)


@jax.jit
def Cl_int_iz_ljz(WT1, WT2l, Pkl, invH, invchi2, weights):
    # for window w/o RSD X window w/ RSD
    return np.einsum("iz,ljz,lz,z,z,z->lij", WT1, WT2l, Pkl, invH, invchi2, weights)


@jax.jit
def Cl_int_liz_ljz(WT1l, WT2l, Pkl, invH, invchi2, weights):
    # for window w/ RSD X window w/ RSD
    return np.einsum("liz,ljz,lz,z,z,z->lij", WT1l, WT2l, Pkl, invH, invchi2, weights)


@jax.jit
def _cosebi_einsum_global(kernel_array, ell_weight, cl_stack):
    # kernel_array: (n_modes, n_ell), ell_weight: (n_ell,), cl_stack: (n_pairs, 2, n_ell)
    # -> (n_pairs, 2, n_modes)
    return np.einsum("ml,l,pql->pqm", kernel_array, ell_weight, cl_stack)


@jax.jit
def _cosebi_einsum_perbin(kernel_array, ell_weight, cl_eb):
    # kernel_array: (n_modes, n_ell), ell_weight: (n_ell,), cl_eb: (2, n_ell)
    # -> (2, n_modes)
    return np.einsum("ml,l,ql->qm", kernel_array, ell_weight, cl_eb)


def _growth_rate_on_grid(perturbations, zs_target):
    # JAX-style backends; to be used for RSD calculation
    try:
        return perturbations.growth_rate(zs_target)
    except TypeError:
        pass

    cache = getattr(perturbations, "_cloelib_growth_rate_cache", None)
    if cache is None:
        f_raw = perturbations.growth_rate()
        z_raw = getattr(perturbations, "z", zs_target)
        perturbations._cloelib_growth_rate_cache = (z_raw, f_raw)
    else:
        z_raw, f_raw = cache

    # If grids match, return directly
    try:
        if (len(z_raw) == len(zs_target)) and (z_raw == zs_target).all():
            return f_raw
    except Exception:
        pass

    return np.interp(zs_target, z_raw, f_raw, left=f_raw[0], right=f_raw[-1])


def _resolve_w_ell(w_ell, bin_key, ns):
    """
    Return ``(kernel_array, thmin, thmax)`` for a given bin pair.

    Supports two calling conventions for ``w_ell``:

    * **Global kernels** — a single dict keyed by integer mode index (and
      ``"metadata"``).  The same kernels are used for every bin pair.
    * **Per-bin kernels** — a dict keyed by bin-pair tuples ``(i, j)``,
      where each value is itself a dict keyed by integer mode index (and
      ``"metadata"``).  The kernel for ``bin_key`` is looked up first; if
      the pair is absent the ``(j, i)`` transpose is tried; if still absent
      the global fallback (key ``None``) is used.

    Parameters
    ----------
    w_ell : dict
        Either a global kernel dict or a per-bin dict of kernel dicts.
    bin_key : tuple
        Full cells key ``('SHE', 'SHE', i, j)``; only the bin indices
        ``(i, j)`` are used for lookup.
    ns : np.ndarray
        Integer mode indices already cast to a numpy array.

    Returns
    -------
    kernel_array : np.ndarray, shape ``(len(ns), n_ell)``
    thmin : float
    thmax : float
    """
    i, j = bin_key[2], bin_key[3]
    # Detect per-bin layout: values are dicts (not arrays)
    first_val = next(v for k, v in w_ell.items() if k != "metadata")
    if isinstance(first_val, dict):
        # Per-bin: try (i,j), then (j,i), then global fallback None
        kernel_dict = w_ell.get((i, j)) or w_ell.get((j, i)) or w_ell.get(None)
        if kernel_dict is None:
            raise KeyError(
                f"No w_ell kernel found for bin pair ({i}, {j}). "
                "Provide either a (i,j)-keyed entry or a None fallback."
            )
    else:
        # Global: the dict itself is the kernel dict
        kernel_dict = w_ell

    kernel_array = np.stack([np.asarray(kernel_dict[int(n)]) for n in ns], axis=0)
    thmin = kernel_dict["metadata"]["THMIN"]
    thmax = kernel_dict["metadata"]["THMAX"]
    return kernel_array, thmin, thmax


def get_cosebis_from_cl(cells, ells, w_ell, ns, software=None):
    """
    Compute EE and BB COSEBIs for all SHE-SHE keys in `cells`.

    Can be used as a standalone function without instantiating `AngularTwoPoint`
    if angular power spectra are already available.

    Parameters
    ----------
    cells : dict
        Angular power spectra in cosmolib format.  All SHE-SHE bin pairs
        present in the dict are processed automatically.
    ells : jax.numpy.ndarray
        Multipoles at which the integration is performed.
    w_ell : dict
        Harmonic-space COSEBIs kernels.  Two layouts are accepted:

        * **Global** — a single dict ``{n: array, ..., "metadata": {...}}``
          (as returned by ``get_W_ell``).  The same kernels are used for
          every bin pair.
        * **Per-bin** — a dict ``{(i, j): {n: array, ..., "metadata": {...}},
          ...}`` supplying independent kernels per bin pair.  A ``None`` key
          may be included as a global fallback for pairs without an explicit
          entry.
    ns : array-like
        Mode indices selecting kernels from `w_ell`.
    software : str, optional
        Software provenance tag stored in the output `COSEBI` objects.
        Defaults to ``'get_cosebis_from_cl (cloelib)'``.

    Returns
    -------
    dict
        Dictionary keyed like the SHE-SHE entries of `cells` with `COSEBI`
        values of shape ``(2, 2, n_modes)``.
    """
    if software is None:
        software = "get_cosebis_from_cl (cloelib)"

    ns = np.asarray(ns)
    # Single device->host sync for nmodes — moved outside any loop
    nmodes = int(np.max(ns))
    n_modes = ns.shape[0]
    # Pre-compute the ell weighting factor once: shape (n_ell,)
    ell_weight = ells * simpsons_weights_jit(len(ells)) / (2 * np.pi)

    she_she = [
        (key, cl_map)
        for key, cl_map in cells.items()
        if key[0] == "SHE" and key[1] == "SHE"
    ]

    if not she_she:
        return {}

    # Detect global vs per-bin kernel layout (mirrors _resolve_w_ell logic)
    first_val = next(v for k, v in w_ell.items() if k != "metadata")
    is_global = not isinstance(first_val, dict)

    tomo_cosebis = {}

    def _interp_cl(cl_map):
        """Interpolate or return EE/BB slices onto `ells`."""
        # Fast path: skip interp when ells is literally the same array object
        # (common when called from get_cosebis which passes the same ells it
        # used to compute the Cls).
        if cl_map.ell is ells:
            return cl_map.array[0, 0], cl_map.array[1, 1]
        return (
            np.interp(ells, cl_map.ell, cl_map.array[0, 0]),
            np.interp(ells, cl_map.ell, cl_map.array[1, 1]),
        )

    if is_global:
        # All pairs share the same kernel: one batched JIT'd einsum for all pairs.
        kernel_array, thmin, thmax = _resolve_w_ell(w_ell, she_she[0][0], ns)
        n_pairs = len(she_she)

        # Stack EE and BB for all pairs: (n_pairs, 2, n_ell)
        cl_stack = np.stack(
            [np.stack(list(_interp_cl(cl_map))) for _, cl_map in she_she]
        )

        # Single JIT'd einsum: (n_pairs, 2, n_modes)
        vals_all = _cosebi_einsum_global(kernel_array, ell_weight, cl_stack)

        # Build all result arrays in 3 batch JAX ops instead of n_pairs*3
        arr_all = np.zeros((n_pairs, 2, 2, n_modes), dtype=np.float64)
        arr_all = arr_all.at[:, 0, 0, :].set(vals_all[:, 0, :])
        arr_all = arr_all.at[:, 1, 1, :].set(vals_all[:, 1, :])

        # Pure Python loop — no JAX ops, no device syncs
        for idx, (key, _) in enumerate(she_she):
            tomo_cosebis[key] = COSEBI(
                array=arr_all[idx],
                mode=ns,
                nmodes=nmodes,
                thmin=thmin,
                thmax=thmax,
                software=software,
            )
    else:
        # Per-bin kernels: one JIT'd einsum per pair over both EE and BB simultaneously
        for key, cl_map in she_she:
            kernel_array, thmin, thmax = _resolve_w_ell(w_ell, key, ns)

            # Stack EE and BB: (2, n_ell)
            cl_eb = np.stack(list(_interp_cl(cl_map)))

            # JIT'd einsum: (2, n_modes)
            vals = _cosebi_einsum_perbin(kernel_array, ell_weight, cl_eb)

            arr = np.zeros((2, 2, n_modes), dtype=np.float64)
            arr = arr.at[0, 0, :].set(vals[0])
            arr = arr.at[1, 1, :].set(vals[1])
            tomo_cosebis[key] = COSEBI(
                array=arr,
                mode=ns,
                nmodes=nmodes,
                thmin=thmin,
                thmax=thmax,
                software=software,
            )

    return tomo_cosebis


def get_cosebis_from_2pcf(twopcf, theta, T_plus, T_minus, ns, software=None):
    """
    Compute EE and BB COSEBIs for all SHE-SHE keys in `twopcf`.

    Can be used as a standalone function without instantiating `AngularTwoPoint`
    if two-point correlation functions are already available.

    Parameters
    ----------
    twopcf : dict
        Two-point correlation functions in cosmolib format.
        Keys should be tuples like ``('SHE', 'SHE', i, j)``.
    theta : jax.numpy.ndarray
        Angular scales in radians.
    T_plus : array-like
        Real-space T_+ kernel functions.
    T_minus : array-like
        Real-space T_- kernel functions.
    ns : jax.numpy.ndarray
        Mode indices selecting kernels from `T_plus`/`T_minus`.
    software : str, optional
        Software provenance tag stored in the output `COSEBI` objects.
        Defaults to ``'get_cosebis_from_2pcf (cloelib)'``.

    Returns
    -------
    dict
        COSEBIs with EE and BB modes, keyed like `twopcf`.
    """
    if software is None:
        software = "get_cosebis_from_2pcf (cloelib)"
    T_plus = np.asarray(T_plus)
    T_minus = np.asarray(T_minus)
    ns = np.asarray(ns)
    tomo_cosebis = {}

    for key, cf_map in twopcf.items():
        if (key[0] == "SHE") & (key[1] == "SHE"):
            continue

        xi_plus = np.interp(theta, cf_map.theta, cf_map.array[0, 0])
        xi_minus = np.interp(theta, cf_map.theta, cf_map.array[1, 1])

        def compute_cosebi(T_p, T_m):
            weights = simpsons_weights_jit(len(theta))
            ee = np.sum(xi_plus * T_p * weights) / np.pi
            bb = np.sum(xi_minus * T_m * weights) / np.pi
            return ee, bb

        ee_vals, bb_vals = jax.vmap(compute_cosebi)(T_plus[ns], T_minus[ns])

        arr = np.zeros((2, 2, ns.shape[0]), dtype=np.float64)
        arr = arr.at[0, 0, :].set(ee_vals)
        arr = arr.at[1, 1, :].set(bb_vals)
        tomo_cosebis[key] = COSEBI(
            array=arr,
            mode=ns,
            nmodes=int(np.max(ns)),
            thmin=np.min(theta),
            thmax=np.max(theta),
            software=software,
        )

    return tomo_cosebis


class AngularTwoPoint:
    """Two point asbtract class to compute two point functions."""

    def __init__(self, tracer1: Tracer = None, tracer2: Tracer = None):
        """
        Initialize the AngularTwoPoint instance.

        Checks if the tracers are compatible and sets the two tracers
        as instance attributes.

        Parameters:
            tracer1 (Tracer, optional): The first tracer for the two-point function. Defaults to None.
            tracer2 (Tracer, optional): The second tracer for the two-point function. Defaults to None.
        """
        self.tracer1 = tracer1
        self.tracer2 = tracer2

    def _software_tag(self, method):
        """
        Standardized software provenance string.

        Parameters
        ----------
        method : callable
            The method generating the data product.

        Returns
        -------
        str
            Software provenance tag.
        """
        return f"{self.__class__.__name__} (cloelib), `{method.__name__}` method"

    def _matter_power_spectrum_limber_grid(
        self, z_l, ks, zs, ells
    ) -> jax.numpy.ndarray:
        """
        Prepare the matter power spectrum grid for Limber approximation.

        It calculates the k values on the Limber grid using the comoving
        distances and multipoles, then interpolates the matter power
        spectrum accordingly.

        Parameters:
            z_l (jax.numpy.ndarray): Redshift grid for Limber integration.
            ks (jax.numpy.ndarray): Wavenumber grid of the matter power spectrum.
            zs (jax.numpy.ndarray): Redshift grid of the matter power spectrum.
            ells (jax.numpy.ndarray): Multipole moments for angular power spectrum.

        Returns:
            (jax.numpy.ndarray): Interpolated matter power spectrum on the Limber grid.
        """
        chi = self.tracer1.perturbations.background.comoving_distance(z_l)
        k_lz = np.expand_dims((ells + 0.5), 1) / chi
        Pk = self.tracer1.perturbations.matter_power_spectrum(zs, ks)
        Pkl = Pkl_interp_vmap(k_lz, z_l, ks, zs, Pk.T)
        return Pkl

    @profile_function
    def get_Cl_tensor(self, ells, nl, ks) -> jax.numpy.ndarray:
        """
        Compute the angular power spectrum Cl using Limber approximation,
        as a plain `jax.numpy.ndarray` - the exact same computation
        `get_Cl` runs, without the final packaging step.

        `get_Cl` itself is now also differentiable via `jax.grad`/
        `jax.jacobian` (as of `cosmolib`'s
        `26-fix-jax-clash-with-cloelib-photo-classes` fix: its
        `AngularPowerSpectrum.__post_init__` used to unconditionally do
        `np.asarray(self.array, dtype=float)` - a plain NumPy cast that
        severs any `jax.grad` trace passing through it
        (`TracerArrayConversionError`) - it now branches on `jax.Array`
        and uses `jax.numpy.asarray` instead when the input is JAX's, so
        it no longer breaks the trace). Prefer `get_Cl_tensor` over
        `get_Cl` anyway when you don't need the packaged, per-pair-type
        `dict`: it skips building that dict and the `AngularPowerSpectrum`
        wrapper objects entirely, which is the more meaningful saving for
        anything running under `jax.jit`/`jax.vmap`. The physics itself
        (Limber integral, window functions, and - with the JAX-native
        cosmology backend - the growth-factor ODE solve and halofit) was
        always fully differentiable; confirmed against finite differences
        in `playground/tutorials/observables/photo_autodiff.ipynb` and
        `validation/photo_autodiff.ipynb`.

        Parameters:
            ells (jax.numpy.ndarray): Multipole moments for the angular power spectrum.
            nl (jax.numpy.ndarray): Noise power spectrum (not used yet, reserved for future use).
            ks (jax.numpy.ndarray): Wavenumber grid of the matter power spectrum.

        Returns:
            (jax.numpy.ndarray): Angular power spectrum Cl for the given multipoles,
            shape `(len(ells), n_bins1, n_bins2)` - `get_Cl`'s packaging
            (`_package_cl`) builds its per-pair-type output (e.g. SHE-SHE's
            2x2 E/B-mode block) from these same values, not a plain
            reshape of this tensor; e.g. for a SHE-SHE pair,
            `get_Cl(...)[("SHE","SHE",i,j)].array[0, 0]` (the EE block)
            equals `get_Cl_tensor(...)[:, i-1, j-1]` exactly.
        """
        contributions1 = getattr(self.tracer1, "get_contributions", lambda: ())()
        contributions2 = getattr(self.tracer2, "get_contributions", lambda: ())()

        if needs_generalized_engine(contributions1, contributions2):
            return self._compute_cl_generalized(
                ells, ks, contributions1, contributions2
            )
        return self._compute_cl_legacy(ells, nl, ks)

    @profile_function
    def get_Cl(self, ells, nl, ks) -> dict:
        """
        Compute the angular power spectrum Cl using Limber approximation.

        Combines the window functions of the tracers, interpolated matter power
        spectrum, Hubble parameter, and comoving distances to calculate the
        two-point angular statistics.

        Differentiable via `jax.grad` (requires `cosmolib`'s
        `26-fix-jax-clash-with-cloelib-photo-classes` fix; see
        `get_Cl_tensor`'s docstring). `get_Cl_tensor` returns the same
        computation without the packaging step, which is still cheaper for
        code that doesn't need the packaged `dict`.

        Parameters:
            ells (jax.numpy.ndarray): Multipole moments for the angular power spectrum.
            nl (jax.numpy.ndarray): Noise power spectrum (not used yet, reserved for future use).
            ks (jax.numpy.ndarray): Wavenumber grid of the matter power spectrum.

        Returns:
            (jax.numpy.ndarray): Angular power spectrum Cl for the given multipoles.
        """
        C_ell_calc = self.get_Cl_tensor(ells, nl, ks)
        self.C_ell_calc = C_ell_calc

        return self._package_cl(C_ell_calc, ells)

    def _compute_cl_legacy(self, ells, nl, ks):
        """Fast path: one shared Pk grid, tracer-level windows.

        Exercised whenever neither tracer's contributions declare any extra
        `SpectrumRequest`s (`spectrum_engine.needs_generalized_engine` is
        `False`) - every configuration that doesn't opt into a
        generalized-engine-aware contribution (e.g. `ia_model="TATT"`).
        Only the packaging at the end is shared with `_compute_cl_generalized`
        (`_package_cl`); the Cl computation itself is fully independent.
        """
        c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s
        zs_calc = self.tracer1.z
        dz = self.tracer1.z[1] - self.tracer1.z[0]
        H = self.tracer1.perturbations.background.hubble_parameter(
            zs_calc, units="km/s/Mpc"
        )
        chi = self.tracer1.perturbations.background.comoving_distance(zs_calc)
        chi2 = chi**2
        WT1 = self.tracer1.get_window(zs_calc)
        WT2 = self.tracer2.get_window(zs_calc)

        # If the two tracers are literally the same object, reuse windows
        same_tracer = self.tracer1 is self.tracer2
        if same_tracer:
            WT2 = WT1

        need_rsd = (
            isinstance(self.tracer1, PositionsTracer)
            and getattr(self.tracer1, "include_rsd", False)
        ) or (
            isinstance(self.tracer2, PositionsTracer)
            and getattr(self.tracer2, "include_rsd", False)
        )

        WT1_rsd = None
        WT2_rsd = None

        if need_rsd:
            f = _growth_rate_on_grid(self.tracer1.perturbations, zs_calc)

            if isinstance(self.tracer1, PositionsTracer) and self.tracer1.include_rsd:
                WT1_rsd = self.tracer1.get_window_rsd(ells, H, f, chi)

            if same_tracer:
                WT2_rsd = WT1_rsd
            else:
                if (
                    isinstance(self.tracer2, PositionsTracer)
                    and self.tracer2.include_rsd
                ):
                    WT2_rsd = self.tracer2.get_window_rsd(ells, H, f, chi)

        Pkl = self._matter_power_spectrum_limber_grid(
            zs_calc, ks, self.tracer1.perturbations.z, ells
        )
        # Added the prefactor here as this is where we have access to ells.
        # There may be a more efficient way to do the multiplication
        prefactor = (
            np.sqrt((ells + 2.0) * (ells + 1.0) * ells * (ells - 1.0))
            / (ells + 0.5) ** 2
        )
        # Did it this way to avoid an if statement, but would be good to know how necessary this is
        prefactor_cell = (
            prefactor * self.tracer1.prefact_toggle + 1 - self.tracer1.prefact_toggle
        ) * (prefactor * self.tracer2.prefact_toggle + 1 - self.tracer2.prefact_toggle)
        weights = simpsons_weights_jit(len(H))

        # C_ell_calc = (
        #    c_0
        #    * Cl_integration(WT1, WT2, Pkl, H, chi2, weights)
        #    * dz
        #    * prefactor_cell[:, None, None]
        # )
        # self.C_ell_calc = C_ell_calc

        invH = 1.0 / H
        invchi2 = 1.0 / chi2

        # --- Base term (no RSD): uses the FAST path always
        C_ell_calc = c_0 * Cl_integration(WT1, WT2, Pkl, H, chi2, weights) * dz

        # --- Add RSD corrections only if needed
        if WT1_rsd is not None:
            # (RSD_1 × dens_2)
            C_ell_calc = (
                C_ell_calc
                + c_0 * Cl_int_liz_jz(WT1_rsd, WT2, Pkl, invH, invchi2, weights) * dz
            )

        if WT2_rsd is not None:
            # (dens_1 × RSD_2)
            C_ell_calc = (
                C_ell_calc
                + c_0 * Cl_int_iz_ljz(WT1, WT2_rsd, Pkl, invH, invchi2, weights) * dz
            )

        if (WT1_rsd is not None) and (WT2_rsd is not None):
            # (RSD_1 × RSD_2)
            C_ell_calc = (
                C_ell_calc
                + c_0
                * Cl_int_liz_ljz(WT1_rsd, WT2_rsd, Pkl, invH, invchi2, weights)
                * dz
            )

        # Apply prefactor as before
        C_ell_calc = C_ell_calc * prefactor_cell[:, None, None]
        return C_ell_calc

    def _compute_cl_generalized(self, ells, ks, contributions1, contributions2):
        """Cl via the per-contribution-pair engine (`spectrum_engine.py`).

        Reached only when some contribution declares extra `SpectrumRequest`s
        (e.g. `TATTContribution`). Sums `Cl_integration(W1, W2, Pkl_pair, ...)`
        over every `(c1, c2)` in the Cartesian product of both tracers'
        contributions, each pair using its own effective P(k,z)
        (`spectrum_engine.get_effective_pk`, falling back to the plain
        matter Pk when a pair has nothing special to say) - the same
        physics separation `toy_cloelib.engine.compute_angular_power_
        spectrum` uses, reusing cloelib's own existing jitted Limber
        kernels (`Pkl_interp_vmap`, `Cl_integration`) unchanged for each
        pair's integral.

        Does not support RSD (`PositionsTracer(..., include_rsd=True)`)
        paired with a generalized-engine-requiring contribution - that
        combination isn't exercised by TATT and is left as a documented gap
        rather than guessed at.
        """
        if (
            isinstance(self.tracer1, PositionsTracer)
            and getattr(self.tracer1, "include_rsd", False)
        ) or (
            isinstance(self.tracer2, PositionsTracer)
            and getattr(self.tracer2, "include_rsd", False)
        ):
            raise NotImplementedError(
                "The generalized Cl engine (contributions declaring extra "
                "SpectrumRequests, e.g. TATTContribution) does not support "
                "PositionsTracer(include_rsd=True) yet."
            )

        c_0 = SPEED_OF_LIGHT / 1000
        zs_calc = self.tracer1.z
        dz = self.tracer1.z[1] - self.tracer1.z[0]
        H = self.tracer1.perturbations.background.hubble_parameter(
            zs_calc, units="km/s/Mpc"
        )
        chi = self.tracer1.perturbations.background.comoving_distance(zs_calc)
        chi2 = chi**2
        weights = simpsons_weights_jit(len(H))

        pert_zs = self.tracer1.perturbations.z
        matter_pk = self.tracer1.perturbations.matter_power_spectrum(pert_zs, ks)
        bank = build_spectra_bank(
            contributions1, contributions2, matter_pk, ks, pert_zs
        )

        k_lz = np.expand_dims((ells + 0.5), 1) / chi
        n_bin1 = self.tracer1.n_z_bins
        n_bin2 = self.tracer2.n_z_bins
        C_ell_calc = np.zeros((len(ells), n_bin1, n_bin2))

        for c1 in contributions1:
            for c2 in contributions2:
                pk_eff = get_effective_pk(c1, c2, bank)
                if pk_eff is None:
                    Pkl_pair = Pkl_interp_vmap(k_lz, zs_calc, ks, pert_zs, matter_pk.T)
                else:
                    # Effective spectra (e.g. TATT's GI/II terms) are
                    # generically signed - see `Pkl_interp_signed`'s
                    # docstring - so they can't go through the log-log
                    # `Pkl_interp` the always-positive matter Pk uses above.
                    Pkl_pair = Pkl_interp_signed_vmap(
                        k_lz, zs_calc, ks, pert_zs, pk_eff.T
                    )

                W1 = c1.compute_kernel(zs_calc)
                W2 = c2.compute_kernel(zs_calc)
                C_ell_calc = C_ell_calc + Cl_integration(
                    W1, W2, Pkl_pair, H, chi2, weights
                )

        C_ell_calc = C_ell_calc * c_0 * dz

        prefactor = (
            np.sqrt((ells + 2.0) * (ells + 1.0) * ells * (ells - 1.0))
            / (ells + 0.5) ** 2
        )
        prefactor_cell = (
            prefactor * self.tracer1.prefact_toggle + 1 - self.tracer1.prefact_toggle
        ) * (prefactor * self.tracer2.prefact_toggle + 1 - self.tracer2.prefact_toggle)
        return C_ell_calc * prefactor_cell[:, None, None]

    def _package_cl(self, C_ell_calc, ells) -> dict:
        """Slice/reshape `C_ell_calc` into the cosmolib-format output dict.

        Shared, untouched packaging logic - identical regardless of which
        computation produced `C_ell_calc`.
        """
        n_bin1 = self.tracer1.n_z_bins
        n_bin2 = self.tracer2.n_z_bins
        C_ell_out = {}

        # Prepare dictionary with tuples as keys for the output
        # This is to match the expected output format of the mixing matrices
        # in the cosmolib format

        # Keep in mind that the output is a dictionary with keys
        # like ('POS', 'POS', i, j) or ('SHE', 'SHE', i, j) where i and j
        # are the bin indices.

        # SHE - SHE returns an array of shape (2, 2, len(ells))
        # Why? Because it expects B-modes. Currently, the B-modes are not implemented,
        # so the second, third and fourth dimensions are filled with zeros.
        # POS - SHE returns an array of shape (2, len(ells))
        # Why? Because it expects the cross-correlation between positions and shear.
        # so the second dimension is filled with zeros.
        # POS - POS returns an array of shape (len(ells))
        # CMBL - SHE returns an array of shape (2, len(ells))
        # Why? Because it expects the cross-correlation between CMB lensing and shear.
        # so the second dimension is filled with zeros.
        # CMBL - POS returns an array of shape (len(ells))
        # CMBL - CMBL returns an array of shape (len(ells))

        def pos_pos_rule(C, i, j):
            return {("POS", "POS", i, j): C[:, i - 1, j - 1]}

        def pos_she_rule(C, i, j):
            block1 = C[:, i - 1, j - 1]
            block2 = C[:, j - 1, i - 1]

            return {
                ("POS", "SHE", i, j): np.stack([block1, np.zeros_like(block1)]),
                ("POS", "SHE", j, i): np.stack([block2, np.zeros_like(block2)]),
            }

        def she_she_rule(C, i, j):
            block = C[:, i - 1, j - 1]
            arr = np.zeros((2, 2, block.shape[0]), dtype=block.dtype)
            arr = arr.at[0, 0, :].set(block)
            return {("SHE", "SHE", i, j): arr}

        def cmbl_cmbl_rule(C, i, j):
            return {("CMBL", "CMBL", i, j): C[:, i - 1, j - 1]}

        def cmbl_pos_rule(C, i, j):
            a, b = sorted((i, j))
            return {("CMBL", "POS", a, b): C[:, i - 1, j - 1]}

        def cmbl_she_rule(C, i, j):
            block = C[:, i - 1, j - 1]
            a, b = sorted((i, j))
            return {("CMBL", "SHE", a, b): np.stack([block, np.zeros_like(block)])}

        tracer_rules = {
            (PositionsTracer, PositionsTracer): pos_pos_rule,
            (PositionsTracer, ShearTracer): pos_she_rule,
            (ShearTracer, ShearTracer): she_she_rule,
            (CMBLensingTracer, PositionsTracer): cmbl_pos_rule,
            (CMBLensingTracer, ShearTracer): cmbl_she_rule,
            (CMBLensingTracer, CMBLensingTracer): cmbl_cmbl_rule,
        }

        # normalize the key so (A, B) and (B, A) are both supported
        key = (type(self.tracer1), type(self.tracer2))
        if key not in tracer_rules and key[::-1] in tracer_rules:
            key = key[::-1]

        rule_fn = tracer_rules.get(key)
        if rule_fn is None:
            raise ValueError(
                f"No rule defined for tracers {type(self.tracer1)}, {type(self.tracer2)}"
            )

        # Vectorized update of C_ell_out using dictionary comprehensions
        a, b = sorted((n_bin1, n_bin2))
        C_ell_out = {
            k: v
            for i in range(1, a + 1)
            for j in range(i, b + 1)
            for k, v in (
                rule_fn(C_ell_calc, i, j)
                if n_bin1 <= n_bin2
                else rule_fn(C_ell_calc, j, i)
            ).items()
        }

        # Use dictionary comprehension for cosmolib_Cls creation
        return {
            key: AngularPowerSpectrum(
                array=array,
                axis=None,
                lower=None,
                upper=None,
                ell=ells,
                software=self._software_tag(self.get_Cl),
            )
            for key, array in C_ell_out.items()
        }

    def get_pseudo_Cl(self, nl, ks, mixing_matrix) -> dict:
        """
        Compute the pseudo angular power spectrum Cl convolved with the mixing matrices.

        Parameters:
        - nl (jax.numpy.ndarray): Noise power spectrum (not used yet).
        - ks (jax.numpy.ndarray): Wavenumber grid of the matter power spectrum.
        - mixing_matrix (dict): Mixing matrices in the euclidlib internal format.

        Returns:
        - dict: Pseudo angular power spectrum Cl for the multipoles specified by the mixing matrix.
        """
        # Determine ellmax from mixing_matrix based on tracer types
        tracer_types = (type(self.tracer1), type(self.tracer2))
        tracer_keys = {
            (PositionsTracer, PositionsTracer): ("POS", "POS"),
            (PositionsTracer, ShearTracer): ("POS", "SHE"),
            (ShearTracer, PositionsTracer): ("POS", "SHE"),
            (ShearTracer, ShearTracer): ("SHE", "SHE"),
        }
        if tracer_types not in tracer_keys:
            raise ValueError("Unsupported tracer pair for mixing matrix.")
        key_type = tracer_keys[tracer_types]
        ellmax = mixing_matrix[key_type + (1, 1)].shape[
            -1
        ]  # mixing matrices will be computed for higher \ell than .upper
        ell = np.arange(0, ellmax)
        # Compute Cls up to ellmax
        C_ell_calc = self.get_Cl(ell, nl, ks)
        n_bin = self.tracer1.n_z_bins
        C_ell_out = {}

        # Helper for POS-SHE symmetry
        def fill_pos_she(i, j):
            for a, b in [(i, j), (j, i)]:
                arr = np.zeros((2, mixing_matrix[("POS", "SHE", a, b)].ell.shape[0]))
                for idx in [0, 1]:
                    arr = arr.at[idx].set(
                        mixing_matrix[("POS", "SHE", a, b)].array
                        @ C_ell_calc[("POS", "SHE", a, b)].array[idx]
                    )
                C_ell_out[("POS", "SHE", a, b)] = arr

        # Main logic for each tracer combination
        if tracer_types == (PositionsTracer, PositionsTracer):
            for i in range(1, n_bin + 1):
                for j in range(i, n_bin + 1):
                    key = ("POS", "POS", i, j)
                    C_ell_out[key] = mixing_matrix[key].array @ C_ell_calc[key].array

        elif tracer_types in [
            (PositionsTracer, ShearTracer),
            (ShearTracer, PositionsTracer),
        ]:
            for i in range(1, n_bin + 1):
                for j in range(i, n_bin + 1):
                    fill_pos_she(i, j)

        elif tracer_types == (ShearTracer, ShearTracer):
            for i in range(1, n_bin + 1):
                for j in range(i, n_bin + 1):
                    key = ("SHE", "SHE", i, j)
                    arr = np.zeros((2, 2, mixing_matrix[key].ell.shape[0]))
                    arr = arr.at[0, 0, :].set(
                        mixing_matrix[key].array[0] @ C_ell_calc[key].array[0, 0]
                        + mixing_matrix[key].array[1] @ C_ell_calc[key].array[1, 1]
                    )

                    arr = arr.at[0, 1, :].set(
                        mixing_matrix[key].array[2] @ C_ell_calc[key].array[0, 1]
                    )

                    arr = arr.at[1, 0, :].set(
                        mixing_matrix[key].array[2] @ C_ell_calc[key].array[1, 0]
                    )

                    arr = arr.at[1, 1, :].set(
                        mixing_matrix[key].array[0] @ C_ell_calc[key].array[1, 1]
                        + mixing_matrix[key].array[1] @ C_ell_calc[key].array[0, 0]
                    )

                    C_ell_out[key] = arr

        # Wrap results in Map objects
        return {
            key: AngularPowerSpectrum(
                array=array,
                axis=None,
                lower=mixing_matrix[key].lower,
                upper=mixing_matrix[key].upper,
                ell=mixing_matrix[key].ell,
                software=self._software_tag(self.get_pseudo_Cl),
            )
            for key, array in C_ell_out.items()
        }

    def get_cosebis(self, ells, nl, ks, w_ell, ns):
        """Compute EE and BB COSEBIs from angular power spectra.

        Delegates to the module-level :func:`get_cosebis_from_cl`. See that
        function for full parameter documentation.
        """
        cells = self.get_Cl(ells, nl, ks)

        return get_cosebis_from_cl(
            cells,
            ells,
            w_ell,
            ns,
            software=self._software_tag(self.get_cosebis),
        )

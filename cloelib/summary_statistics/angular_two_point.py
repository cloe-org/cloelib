"""Module for angular two-point functions."""

# cloelib imports
from cloelib.observables.tracer import Tracer
from cloelib.observables.photo import PositionsTracer
from cloelib.observables.photo import ShearTracer
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
    def get_Cl(self, ells, nl, ks) -> dict:
        """
        Compute the angular power spectrum Cl using Limber approximation.

        Combines the window functions of the tracers, interpolated matter power
        spectrum, Hubble parameter, and comoving distances to calculate the
        two-point angular statistics.

        Parameters:
            ells (jax.numpy.ndarray): Multipole moments for the angular power spectrum.
            nl (jax.numpy.ndarray): Noise power spectrum (not used yet, reserved for future use).
            ks (jax.numpy.ndarray): Wavenumber grid of the matter power spectrum.

        Returns:
            (jax.numpy.ndarray): Angular power spectrum Cl for the given multipoles.
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
        self.C_ell_calc = C_ell_calc

        n_bin = self.tracer1.n_z_bins
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

        tracer_rules = {
            (PositionsTracer, PositionsTracer): pos_pos_rule,
            (PositionsTracer, ShearTracer): pos_she_rule,
            (ShearTracer, ShearTracer): she_she_rule,
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
        C_ell_out = {
            k: v
            for i in range(1, n_bin + 1)
            for j in range(i, n_bin + 1)
            for k, v in rule_fn(C_ell_calc, i, j).items()
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
                        mixing_matrix[("POS", "SHE", a, b)]
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

    def get_cosebis_from_cl(self, cells, ells, w_ell, ns):
        """
        Compute EE and BB COSEBIs for all SHE-SHE keys in `cells`.

        Parameters:
        - cells (dict): Angular power spectrum from AngularTwoPoint get_Cl or get_pseudo_Cl, or, alternatively, Cl calculation from other sources, as long as it is in the same cosmolib format. The keys should be tuples like ('SHE', 'SHE', i, j) where i and j are bin indices.
        - ells (jax.numpy.ndarray): Array with the multipole moments.
        - w_ell (np.array): The kernel functions, obtained via get_W_ell.
        - ns (jax.numpy.ndarray): The indices for the kernel function.

        Returns:
        - dict: COSEBIs obtained from the angular power spectrum with EE and BB modes.
        """
        # Handle 0-D object arrays from np.savez/np.load
        w_ell = np.asarray(w_ell)
        ns = np.asarray(ns)

        tomo_cosebis = {}

        for key, cl_map in cells.items():
            # Filter using dictionary structure rather than if-statements
            (key[0] == "SHE") & (key[1] == "SHE")

            cl_ee = np.interp(ells, cl_map.ell, cl_map.array[0, 0])
            cl_bb = np.interp(ells, cl_map.ell, cl_map.array[1, 1])

            # Vectorized computation using vmap
            def compute_cosebi(w_n):
                weights = simpsons_weights_jit(len(ells))
                ee = np.sum(ells * cl_ee * w_n * weights) / (2 * np.pi)
                bb = np.sum(ells * cl_bb * w_n * weights) / (2 * np.pi)
                return ee, bb

            ee_vals, bb_vals = jax.vmap(compute_cosebi)(w_ell[ns])

            arr = np.zeros((2, 2, ns.shape[0]), dtype=np.float64)
            arr = arr.at[0, 0, :].set(ee_vals)
            arr = arr.at[1, 1, :].set(bb_vals)
            tomo_cosebis[key] = COSEBI(
                array=arr,
                mode=ns,
                nmodes=int(np.max(ns)),
                software=self._software_tag(self.get_cosebis_from_cl),
            )

        return tomo_cosebis

    def get_cosebis_from_2pcf(self, twopcf, theta, T_plus, T_minus, ns):
        """
        Compute EE and BB COSEBIs for all SHE-SHE keys in `twopcf`.

        Parameters:
        - twopcf (dict): Two-point correlation function from AngularTwoPoint and/or other sources, in cosmolib format.
          The keys should be tuples like ('SHE', 'SHE', i, j) where i and j are bin indices.
        - theta (jax.numpy.ndarray): Array with the angular scales in radians.
        - T_plus (np.array): The T_+ kernel functions in real space.
        - T_minus (np.array): The T_- kernel functions in real space.
        - ns (jax.numpy.ndarray): The indices for the kernel function.

        Returns:
        - dict: COSEBIs obtained from the two-point correlation function with EE and BB modes.
        """
        T_plus = np.asarray(T_plus)
        T_minus = np.asarray(T_minus)
        ns = np.asarray(ns)

        tomo_cosebis = {}

        for key, cf_map in twopcf.items():
            (key[0] == "SHE") & (key[1] == "SHE")

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
                software=self._software_tag(self.get_cosebis_from_2pcf),
            )

        return tomo_cosebis

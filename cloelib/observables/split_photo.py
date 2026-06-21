"""
Module with two classes for each observable tracer type: shear and galaxy positions.

Both classes are compatible with the Tracer protocol.
"""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations,Background
from cloelib.auxiliary.math_utils import cached_stacked_simpson, simps
from cloelib.auxiliary.systematics import shift_dndz_jax, stretch_dndz_jax
from cloelib.cosmology.split_cosmology import SplitLinearPerturbations

# General imports
import jax.numpy as np  # type: ignore
import jax  # type: ignore
import interpax  # type: ignore
import jax.lax as lx


# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s


@jax.jit
def _L_coeffs(ells):
    ell = ells.astype(np.float64)
    # Avoid invalid sqrt for ell<2 by using a safe ell in the algebra
    ell_s = np.maximum(ell, 2.0)

    Lm1 = (
        -ell_s
        * (ell_s - 1.0)
        / ((2.0 * ell_s - 1.0) * np.sqrt((2.0 * ell_s - 3.0) * (2.0 * ell_s + 1.0)))
    )
    L0 = (2.0 * ell_s**2 + 2.0 * ell_s - 1.0) / (
        (2.0 * ell_s - 1.0) * (2.0 * ell_s + 3.0)
    )
    Lp1 = (
        -(ell_s + 1.0)
        * (ell_s + 2.0)
        / ((2.0 * ell_s + 3.0) * np.sqrt((2.0 * ell_s + 1.0) * (2.0 * ell_s + 5.0)))
    )

    mask = ell >= 2.0
    Lm1 = np.where(mask, Lm1, 0.0)
    L0 = np.where(mask, L0, 0.0)
    Lp1 = np.where(mask, Lp1, 0.0)
    return Lm1, L0, Lp1


@jax.jit
def _alpha_coeffs(ells):
    ell = ells.astype(np.float64)
    denom = 2.0 * ell + 1.0
    am1 = (2.0 * ell - 3.0) / denom
    a0 = np.ones_like(ell)
    ap1 = (2.0 * ell + 5.0) / denom

    # For ell<2 the L's are zero anyway; keep alphas harmless.
    mask = ell >= 2.0
    am1 = np.where(mask, am1, 1.0)
    ap1 = np.where(mask, ap1, 1.0)
    return am1, a0, ap1


@jax.jit
def _interp_linear_2d_queries(chi, y_bz, xq_lz):
    """
    Linear 2D interpolator for the RSD window

    """
    Z = chi.shape[0]
    idx = np.searchsorted(chi, xq_lz, side="right") - 1
    idx = np.clip(idx, 0, Z - 2)

    idx0 = idx[None, :, :]
    idx1 = (idx + 1)[None, :, :]

    y0 = np.take_along_axis(y_bz[:, None, :], idx0, axis=2)
    y1 = np.take_along_axis(y_bz[:, None, :], idx1, axis=2)

    x0 = np.take_along_axis(chi[None, :], idx, axis=1)
    x1 = np.take_along_axis(chi[None, :], idx + 1, axis=1)

    t = (xq_lz - x0) / (x1 - x0)
    t = t[None, :, :]

    out = y0 + t * (y1 - y0)

    # zero outside bounds
    oob = (xq_lz < chi[0]) | (xq_lz > chi[-1])
    out = np.where(oob[None, :, :], 0.0, out)

    return np.transpose(out, (1, 0, 2))


@jax.jit
def get_photo_rsd(ells, chi, S_bin_z):
    Lm1, L0, Lp1 = _L_coeffs(ells)
    am1, _, ap1 = _alpha_coeffs(ells)

    chi_q_m1 = am1[:, None] * chi[None, :]
    chi_q_p1 = ap1[:, None] * chi[None, :]

    S_m1 = _interp_linear_2d_queries(chi, S_bin_z, chi_q_m1)
    S_0 = np.broadcast_to(S_bin_z[None, :, :], S_m1.shape)
    S_p1 = _interp_linear_2d_queries(chi, S_bin_z, chi_q_p1)

    return (
        Lm1[:, None, None] * S_m1 + L0[:, None, None] * S_0 + Lp1[:, None, None] * S_p1
    )


class ShearTracer:
    """Class for the kernel for Cosmic Shear."""

    def __init__(
        self,
        perturbations: Perturbations,
        background: Background,
        splitlinearperturbations: SplitLinearPerturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        nuisance_params: dict,
        Omega_m_lens: float,
    ):
        r"""
        Initialize the class instance.

        Args:
          perturbations (object): An object from NonLinearPerturbations class
          dndz (np.ndarray): A n-dimensional array representing the number density distribution of galaxies as a function of redshift.
            It is expected to be normalised.
          z (np.ndarray): A 1-dimensional array representing the redshift values corresponding to the `dndz` array.
          Omega_m_IA: Omega matter used as input to intrinsic alignement window. It has to be Omega matter growth.
          Omega_m_lens: Omega matter used as input to the lensing window. It can be either Omega matter growth or geometry.
        """
        if 0.0 in z:
            raise ValueError(
                "One of the z array elements is equal to zero, breaking Limber integration."
            )
        self.perturbations = perturbations
        self.background = background
        self.splitlinearperturbations = splitlinearperturbations
        self.z = z
        self.nuisance_params = nuisance_params
        self.Omega_m_IA = self.perturbations.background.Omega_m(0.0)
        self.Omega_m_lens = Omega_m_lens
        # This is to add the necessary prefactor to shear, while avoiding it in GC
        self.prefact_toggle = 1
        # Set multiplicative bias (m_bias)
        self.m_bias = [
            self.nuisance_params[f"multiplicative_bias_{i + 1}"]
            for i in range(dndz.shape[0])
        ]
        self.dz_shear_i = [
            self.nuisance_params[f"dz_shear_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.width_shear_i = [
            self.nuisance_params[f"width_shear_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.n_z_bins = dndz.shape[0]
        self.dndz = dndz
        # Correct dndz for width_shear
        self.dndz_stretched = stretch_dndz_jax(dndz, z, self.width_shear_i)
        # Correct dndz_stretched for dz_shear
        self.dndz_shifted = shift_dndz_jax(self.dndz_stretched, z, self.dz_shear_i)

    def get_window_IA(self, z):
        r"""Window integrand.

        Calculates IA window

        Args:
          z (float): Redshift at which kernel is being evaluated

        Returns:
          window_IA (np.ndarray):
        """
        Omega_m0 = self.Omega_m_IA
        Hz = self.background.hubble_parameter(z)
        Dz = self.splitlinearperturbations.growth_factor(z, self.perturbations.k)[:, 1]
        # TODO discuss whether we want growth factor to output a 1D or a 2D array
        A_IA = self.nuisance_params["AIA"]
        C_IA = self.nuisance_params["CIA"]
        Eta_IA = self.nuisance_params["EtaIA"]
        factor = -Hz / c_0 * A_IA * C_IA * Omega_m0 * (1 + z) ** Eta_IA / Dz
        return np.einsum("ij, j->ij", self.dndz_shifted, factor)

    def get_lensing_efficiency_bin(self, z, bin_idx):
        """Compute the lensing efficiency in a redshift bin."""
        interpolator = interpax.Akima1DInterpolator(
            self.z, self.dndz_shifted[bin_idx, :]
        )
        x = np.linspace(0.0, 4, 200)
        y = self.background.comoving_distance(x)
        rx_interp = interpax.Akima1DInterpolator(x, y)
        f1 = jax.jit(lambda x: interpolator(x))
        f2 = jax.jit(lambda x: interpolator(x) / rx_interp(x))
        integral_1 = simps(f1, z, 3.0)
        integral_2 = simps(f2, z, 3.0)
        efficiency = integral_1 - integral_2 * self.background.comoving_distance(z)
        return efficiency

    def get_lensing_efficiency(self, z):
        r"""
        Compute the lensing efficiency kernel for each redshift bin.

        This function calculates the geometric lensing kernel W(χ), which weights the contribution
        of matter at different redshifts to the weak lensing signal, for a given redshift grid `z`.

        Args:
          z (np.ndarray): 1D array of redshift values (must be evenly spaced). Used to compute comoving distances
            and define integration domain.

        Returns:
          (np.ndarray): 2D array of shape (N_bins, len(z)) representing the lensing efficiency kernel W(z)
            for each redshift bin over the evaluation grid.

        Notes
        -----
        - Assumes `z` is evenly spaced; spacing is inferred as `z[1] - z[0]`.
        - Uses a precomputed Simpson rule weight matrix (`cached_stacked_simpson`) for integration.
        - `self.dndz` is expected to have shape (N_bins, len(z)) and be normalized.
        - Efficiency is evaluated using `np.einsum`.
        """
        dz = z[1] - z[0]  # assuming equispaced!
        rz = self.background.comoving_distance(z)
        rzrz = 1 - np.outer(rz, 1 / rz)
        w_matrix = cached_stacked_simpson(len(z))
        result = np.einsum("ik, jk, jk->ij", self.dndz_shifted, rzrz, w_matrix) * dz
        return result

    def get_window_lensing(self, z):
        r"""Weak Lensing shear kernel.

        Calculates the weak lensing shear kernel for a given tomographic bin
        distribution.
        Uses broadcasting to compute a 2D-array of integrands and then applies
        `np.trapz` on the array along one axis.

        $$
            W_{i}^{\gamma}(\ell, z, k) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} (1 + z)
            f_K\left[\tilde{r}(z)\right]
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm L}(z^{\prime})
            \frac{f_K\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}}\\
        $$

        Args:
          z (numpy.ndarray): Redshift at which weight is evaluated (`float` type).

        Returns:
          (numpy.ndarray): 1-D Numpy array of shear kernel values for specified bin
            at specified scale for the redshifts defined in z
        """
        Omega_m0 = self.Omega_m_lens
        factor = (
            3
            / 2
            * (self.background.H0 / c_0) ** 2
            * Omega_m0
            * (1 + z)
            * self.background.comoving_distance(z)
        )
        efficiency = self.get_lensing_efficiency(z)
        return np.einsum("ij, j->ij", efficiency, factor)

    def get_window(self, z):
        r"""Compute the Window.

        Computes general window given the selected tracer

        Parameters:
          z (float): Redshift at which window kernel is being evaluated

        Returns:
          window (np.ndarray):
        """
        total_window = self.get_window_lensing(z) + self.get_window_IA(z)
        # Apply multiplicative bias
        total_window *= 1 + np.array(self.m_bias)[:, None]
        return total_window


class PositionsTracer:
    """Class to define the kernel for angular (galaxy) clustering."""

    def __init__(
        self,
        perturbations: Perturbations,
        background = Background,
        dndz: np.ndarray,
        z: np.ndarray,
        galaxy_bias_model: str,
        nuisance_params: dict,
        include_rsd: bool = False,
    ):
        r"""
        Initialize the class instance.

        ### This docstring should be checked. I replaced LinearPerturbations or NonLinearPerturbations with Perturbations as the type of perturbations in the parameters list doc.

        Parameters:
          perturbations (Perturbations): An object from NonLinearPerturbations class
          dndz (np.ndarray): A n-dimensional array representing the number density distribution of galaxies as a function of redshift.
            It is expected to be normalised.
          z (np.ndarray): A 1-dimensional array representing the redshift values corresponding to the `dndz` array.
          galaxy_bias_model (str): A string specifying the model used to describe the galaxy bias
          nuisance_params (dict): A dictionary containing additional parameters that are not directly related to the cosmological model but may affect the observations.
        """
        if 0.0 in z:
            raise ValueError(
                "One of the z array elements is equal to zero, breaking Limber integration."
            )
        self.perturbations = perturbations
        self.background = background
        self.z = z
        # This is to add the necessary prefactor to shear, while avoiding it in GC
        self.prefact_toggle = 0

        self.nuisance_params = nuisance_params
        self.dz_pos_i = [
            self.nuisance_params[f"dz_pos_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.width_pos_i = [
            self.nuisance_params[f"width_pos_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.dndz = dndz
        # Correct dndz for width_pos
        self.dndz_stretched = stretch_dndz_jax(dndz, z, self.width_pos_i)
        # Correct dndz for dz_pos
        self.dndz_shifted = shift_dndz_jax(self.dndz_stretched, z, self.dz_pos_i)
        self.flags = {"galaxy_bias_model": galaxy_bias_model}
        self.n_z_bins = dndz.shape[0]
        self.magnification_bias = [
            self.nuisance_params[f"magnification_bias_{i + 1}"]
            for i in range(dndz.shape[0])
        ]
        self.include_rsd = include_rsd

        # Using dict.get so I can provide a default since lax has to compile every branch of the conditional
        def per_bin_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_photo_bin%d" % bin, 1.0)
                    for bin in range(self.n_z_bins)
                ]
            )
            # lax required same size for all cases, so padding here and will only use first n_z_bins values later
            return np.pad(bias_array, (0, self.z.shape[0] - self.n_z_bins))

        def per_bin_int_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_photo_bin%d" % bin, 1.0)
                    for bin in range(self.n_z_bins)
                ]
            )
            index_max_nz = np.argmax(dndz, axis=1)
            z_nz_max = jax.vmap(
                lambda i: lx.dynamic_index_in_dim(self.z, i, keepdims=False)
            )(index_max_nz)
            return interpax.interp1d(self.z, z_nz_max, bias_array, extrap=True)

        def poly_case():
            poly_order = 3
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_photo_poly%d" % bin, 1.0)
                    for bin in range(poly_order + 1)
                ]
            )
            return (
                bias_array[0]
                + bias_array[1] * z
                + bias_array[2] * z**2
                + bias_array[3] * z**3
            )

        conditions = np.array(
            [
                self.flags["galaxy_bias_model"] == "per_bin",
                self.flags["galaxy_bias_model"] == "per_bin_int",
                self.flags["galaxy_bias_model"] == "poly",
            ]
        )
        index = np.argwhere(conditions, size=1).squeeze()

        self.bias_array = [per_bin_case, per_bin_int_case, poly_case][index]()

    def get_window_positions(self, z) -> np.ndarray:
        r"""Galaxy Positions window function.

        Implements the galaxy clustering photometric window function.

        $$
            W_i^G(z) = \frac{n_i(z)}{\bar{n_i}}\frac{H(z)}{c}\\
        $$

        Parameters:
          z (numpy.ndarray|float): Redshift at which to evaluate distribution (array of `float` or `float`)

        Returns:
          window_positions (numpy.ndarray): Window function for angular photometric galaxy clustering
        """

        def per_bin_case():
            window = (
                self.bias_array[: self.n_z_bins, None]
                * self.dndz_shifted
                * self.background.hubble_parameter(z)
                / c_0
            )
            return window

        def z_func_case():
            window = (
                self.bias_array[None, :]
                * self.dndz_shifted
                * self.background.hubble_parameter(z)
                / c_0
            )
            return window

        conditions = np.array(
            [
                self.flags["galaxy_bias_model"] == "per_bin",
                self.flags["galaxy_bias_model"] in ["per_bin_int", "poly"],
            ]
        )
        index = np.argwhere(conditions, size=1).squeeze()

        window_positions = [per_bin_case, z_func_case][index]()

        return window_positions

    def get_window_rsd(self, ells, H, f, chi) -> np.ndarray:
        r"""
        Linear photo-RSD window :math:`W^{\rm RSD}_i(\ell,z)` tabulated on the internal z-grid.

        We use the shifted-distance (extended Limber) approximation, where the RSD contribution
        is written as a linear combination of the source term evaluated at shifted redshifts:

        $$
        W^{\rm RSD}_i(\ell,z)
        =
        A_\ell\,S_i(z)
        +B_\ell\,S_i\!\left(z_{-2}(\ell,z)\right)
        +C_\ell\,S_i\!\left(z_{+2}(\ell,z)\right),
        $$

        with
        $$
        S_i(z)=\frac{H(z)\,f(z)}{c}\,n_i(z),
        $$

        and the shifted redshifts defined implicitly via comoving distance (here :math:`\chi \equiv r`):
        $$
        \chi\!\left(z_m(\ell,z)\right)=
        \frac{\ell+m+\tfrac12}{\ell+\tfrac12}\,\chi(z),
        \qquad m\in\{-2,+2\}.
        $$

        The coefficients are
        $$
        A_\ell=\frac{2\ell^2+2\ell-1}{(2\ell-1)(2\ell+3)},\quad
        B_\ell=-\frac{\ell(\ell-1)}{(2\ell-1)(2\ell+1)},\quad
        C_\ell=-\frac{(\ell+1)(\ell+2)}{(2\ell+1)(2\ell+3)}.
        $$

        Notes
        -----
        - `chi` is used purely as an interpolation coordinate so `get_photo_rsd` can evaluate
          the shifted arguments efficiently.

        Parameters
        ----------
        ells : array_like
            Multipoles ell.
        H, f : array_like
            H(z) and growth rate f(z), sampled on the same z-grid as `self.dndz_shifted`.
        chi : array_like
            Comoving distance χ(z)=r(z), sampled on the same z-grid.

        Returns
        -------
        ndarray
            The RSD window sampled on the z-grid (shape as returned by `get_photo_rsd`).
        """
        # S_i(z) = H(z) f(z) n_i(z) / c
        S = (H[None, :] * f[None, :] / c_0) * self.dndz_shifted
        return get_photo_rsd(ells, chi, S)

    def get_magnification_efficiency(self, z):
        r"""
        Compute the magnification efficiency kernel for each redshift bin.

        This function calculates the geometric lensing kernel W(χ), which weights the contribution
        of matter at different redshifts to the weak lensing signal, for a given redshift grid `z`.

        Parameters:
          z (np.ndarray): 1D array of redshift values (must be evenly spaced). Used to compute comoving distances
            and define integration domain.

        Returns:
          (np.ndarray): 2D array of shape (N_bins, len(z)) representing the lensing efficiency kernel W(z)
            for each redshift bin over the evaluation grid.

        Notes
        -----
        - Assumes `z` is evenly spaced; spacing is inferred as `z[1] - z[0]`.
        - Uses a precomputed Simpson rule weight matrix (`cached_stacked_simpson`) for integration.
        - `self.dndz` is expected to have shape (N_bins, len(z)) and be normalized.
        - Efficiency is evaluated using `np.einsum`.
        """
        dz = z[1] - z[0]  # assuming equispaced!
        rz = self.background.comoving_distance(z)
        rzrz = 1 - np.outer(rz, 1 / rz)
        w_matrix = cached_stacked_simpson(len(z))
        result = np.einsum("ik, jk, jk->ij", self.dndz_shifted, rzrz, w_matrix) * dz
        return result

    def get_window_magnification(self, z):
        r"""Magnification photometric galaxy kernel.

        Calculates the weak lensing shear kernel for a given tomographic bin
        distribution.
        Uses broadcasting to compute a 2D-array of integrands and then applies
        `np.trapz` on the array along one axis.

        $$
            W_{i}^{\gamma}(\ell, z, k) =
            \frac{3}{2}\left ( \frac{H_0}{c}\right )^2
            \Omega_{{\rm m},0} b_{\rm mag, i} (1 + z)
            f_K\left[\tilde{r}(z)\right]
            \int_{z}^{z_{\rm max}}{{\rm d}z^{\prime} n_{i}^{\rm L}(z^{\prime})
            \frac{f_K\left[\tilde{r}(z^{\prime}) - \tilde{r}(z)\right]}
            {f_K\left[\tilde{r}(z^{\prime})\right]}}\\
        $$

        Parameters:
          z (numpy.ndarray): Redshift at which weight is evaluated (array of `float`).

        Returns:
          (numpy.ndarray): 1-D Numpy array of shear kernel values for specified bin
            at specified scale for the redshifts defined in z
        """
        Omega_m0 = self.Omega_m_lens
        factor = (
            3
            / 2
            * (self.background.H0 / c_0) ** 2
            * Omega_m0
            * (1 + z)
            * self.background.comoving_distance(z)
        )
        efficiency = self.get_magnification_efficiency(z)
        return (
            np.einsum("ij, j->ij", efficiency, factor)
            * np.array(self.magnification_bias)[:, None]
        )

    def get_window(self, z) -> np.ndarray:
        """
        Compute the angular photometric galaxy clustering window function.

        This function combines the galaxy clustering window and the magnification
        bias window to produce the final window function.

        Parameters:
          z (float): Redshift at which window kernel is being evaluated

        Returns:
          window (np.ndarray):
        """
        window = self.get_window_positions(z) + self.get_window_magnification(z)
        return window

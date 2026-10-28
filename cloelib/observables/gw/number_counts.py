"""GW number-count tracer: `GWNumberCountsTracer` and its Contributions.

Compatible with the Tracer protocol. Counterpart to `gw.weak_lensing`, which
holds `GWWeakLensingTracer`.
"""

# cloelib imports
from cloelib.utils.units import SPEED_OF_LIGHT
from cloelib.utils.redshift import check_nonzero_redshifts
from cloelib.cosmology.cosmology import Perturbations
from cloelib.utils.systematics import shift_dndz_jax, stretch_dndz_jax

# General imports
import jax
import jax.numpy as np
import interpax
import jax.lax as lx

# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s


class GWSourceBiasContribution:
    """Source-bias-weighted kernel term of `GWNumberCountsTracer.get_window()`.

    Currently one of the three linear-bias models selected by
    `gw_bias_model` (`GWNumberCountsTracer.get_window_number_counts`).
    """

    def __init__(self, tracer: "GWNumberCountsTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z):
        return self._tracer.get_window_number_counts(z)


class GWNumberCountsTracer:
    """Tracer for the angular number density of GW sources."""

    def __init__(
        self,
        perturbations: Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        gw_bias_model: str,
        nuisance_params: dict,
    ):
        r"""
        Initialize the class instance.

        Parameters:
          perturbations (Perturbations): Perturbation backend providing a
            compatible background cosmology.
          dndz (np.ndarray): A n-dimensional array representing the number density distribution of GW sources as a function of redshift, with shape
            `(n_bins, n_z)`. It is expected to be normalised.
          z (np.ndarray): Evenly sampled, non-zero redshift grid with shape
            `(n_z,)` corresponding to the last axis of `dndz`.
          gw_bias_model (str): A string specifying the model used to describe
            the GW source bias.
          nuisance_params (dict): Redshift-shift and width parameters for every
            bin, plus parameters for the selected GW bias model.
        """
        check_nonzero_redshifts(z)
        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.z = z
        # AngularTwoPoint field responses: scalar number counts need neither
        # the spin-2 shear response nor the GW weak-lensing response.
        self.prefact_toggle = 0
        self.gw_prefact_toggle = 0

        self.nuisance_params = nuisance_params
        self.dz_gw_i = [
            self.nuisance_params[f"dz_gw_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.width_gw_i = [
            self.nuisance_params[f"width_gw_{i + 1}"] for i in range(dndz.shape[0])
        ]
        self.dndz = dndz
        # Correct dndz for width_gw.
        self.dndz_stretched = stretch_dndz_jax(dndz, z, self.width_gw_i)
        # Correct dndz_stretched for dz_gw.
        self.dndz_shifted = shift_dndz_jax(self.dndz_stretched, z, self.dz_gw_i)
        self.flags = {"gw_bias_model": gw_bias_model}
        self.n_z_bins = dndz.shape[0]

        # Use the same bias-model structure and defaults as PositionsTracer.
        def per_bin_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_gw_bin%d" % bin, 1.0)
                    for bin in range(self.n_z_bins)
                ]
            )
            # lax required same size for all cases, so padding here and will only use first n_z_bins values later
            return np.pad(bias_array, (0, self.z.shape[0] - self.n_z_bins))

        def per_bin_int_case():
            bias_array = np.asarray(
                [
                    nuisance_params.get("b1_gw_bin%d" % bin, 1.0)
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
                    nuisance_params.get("b1_gw_poly%d" % bin, 1.0)
                    for bin in range(poly_order + 1)
                ]
            )
            return (
                bias_array[0]
                + bias_array[1] * z
                + bias_array[2] * z**2
                + bias_array[3] * z**3
            )

        # Plain Python dispatch on the static model string: a `jnp`-computed
        # index would be a tracer under `jax.jit` and can't index a list.
        bias_cases = {
            "per_bin": per_bin_case,
            "per_bin_int": per_bin_int_case,
            "poly": poly_case,
        }
        if gw_bias_model not in bias_cases:
            raise ValueError(
                f"Unknown gw_bias_model {gw_bias_model!r}; expected one of "
                f"{sorted(bias_cases)}."
            )
        self.bias_array = bias_cases[gw_bias_model]()
        self.bias = GWSourceBiasContribution(self)

    def get_contributions(self):
        """Return this tracer's window as its separable Contribution terms.

        The contribution currently delegates to `get_window_number_counts`.

        Returns:
          contributions (tuple): `(self.bias,)`.
        """
        return (self.bias,)

    def get_window_number_counts(self, z) -> np.ndarray:
        r"""GW number-count window function.

        Implements the GW number-count source-density window, analogous to
        `PositionsTracer.get_window_positions`, with the galaxy bias replaced
        by the GW source bias.

        $$
            W_i^{\rm GW-NC}(z) =
            b_i^{\rm GW}(z)\,n_i^{\rm GW}(z)\frac{H(z)}{c}
        $$

        Parameters:
          z (numpy.ndarray): Redshift grid at which to evaluate the window.

        Returns:
          window_number_counts (np.ndarray): Angular GW number-count windows
            with shape `(n_bins, n_z)`.
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

        # Static Python branch (see `__init__`), safe under `jax.jit`.
        if self.flags["gw_bias_model"] == "per_bin":
            return per_bin_case()
        return z_func_case()

    def get_window(self, z) -> np.ndarray:
        """
        Compute the angular GW number-count window function.

        Parameters:
          z (np.ndarray): Redshift grid at which the window is evaluated.

        Returns:
          window (np.ndarray): Number-count windows with shape
            `(n_bins, n_z)`.
        """
        total_window = sum(c.compute_kernel(z) for c in self.get_contributions())
        return total_window

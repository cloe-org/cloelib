"""
Module with two classes for each observable tracer type: shear and galaxy positions.

Both classes are compatible with the Tracer protocol.
"""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.Weyl_cosmology import Weyl_Perturbations
from cloelib.observables.photo import PositionsTracer, get_photo_rsd

# General imports
import jax.numpy as np  # type: ignore


# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s

# NOTE WEYL PROJECT: No changes to ShearTracer class --> import from usual photo.py file


class PositionsTracer_Weyl_GC(PositionsTracer):
    """Positions tracer for galaxy clustering including Weyl potential (subclass of PositionsTracer)."""

    def __init__(
        self,
        perturbations: Weyl_Perturbations,  # Note: We require this to be an instance of Weyl_perturbations
        dndz: np.ndarray,
        z: np.ndarray,
        nuisance_params: dict,
        include_rsd: bool = False,
    ):
        # Reuse parent initialization but force per-bin bias model
        super().__init__(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            galaxy_bias_model="per_bin",
            nuisance_params=nuisance_params,
            include_rsd=include_rsd,
        )

        # Defines z_ini
        self.z_ini = self.perturbations.z_ini  # We can directly access z_ini from the Weyl_Perturbations instance, which is set at initialization of that class.

        # Get sigma8 at z_ini
        self.sigma8_ini = self.perturbations.sigma8_zini()

        # Override bias_array to use bhat_binN naming (bhat = b(z)*sigma8(z))
        bias_vals = np.asarray(
            [nuisance_params["bhat_bin%d" % bin] for bin in range(self.n_z_bins)]
        )
        self.bias_array = np.pad(bias_vals, (0, self.z.shape[0] - self.n_z_bins))

    def growth_since_zini(self, z) -> np.ndarray:
        """New function to account for the growth (in GR) since z_ini"""
        growth = (
            self.perturbations.growth_factor(z, self.perturbations.k[:1])[:, 0]
            / self.perturbations.growth_factor(
                np.array([self.z_ini]), self.perturbations.k[:1]
            )[0, 0]
        )
        return growth

    def get_window_positions(self, z) -> np.ndarray:
        """Weyl GC positions window: uses bhat and divides by sigma8_ini (single power)."""

        def per_bin_case():
            window = (
                self.bias_array[: self.n_z_bins, None]
                * self.dndz_shifted
                * self.perturbations.background.hubble_parameter(z)
                / c_0
                / self.sigma8_ini
            )
            return window

        return per_bin_case()

    def get_window_rsd(self, ells, H, f, chi) -> np.ndarray:
        """Weyl GC RSD window: multiply by growth_factor (normalized to z_ini) once."""
        # S_i(z) = H(z) f(z) n_i(z) / c
        growth_factor = self.growth_since_zini(self.z)
        S = (H[None, :] * f[None, :] / c_0) * self.dndz_shifted * growth_factor
        return get_photo_rsd(ells, chi, S)

    def get_window_magnification(self, z):
        """Weyl GC magnification: multiply by growth_factor (normalized to z_ini) once."""
        growth_factor = self.growth_since_zini(z)

        Omega_m0 = self.background.Omega_m(0.0)
        factor = (
            3
            / 2
            * growth_factor
            * (self.background.H0 / c_0) ** 2
            * Omega_m0
            * (1 + z)
            * self.background.comoving_distance(z)
        )
        efficiency = self.get_magnification_efficiency(z)  # inherited from parent
        return (
            np.einsum("ij, j->ij", efficiency, factor)
            * np.array(self.magnification_bias)[:, None]
        )

    # get_magnification_efficiency and get_window are inherited unchanged from PositionsTracer


class PositionsTracer_Weyl_GGL(PositionsTracer):
    """Positions tracer for galaxy-galaxy lensing including Weyl potential (subclass of PositionsTracer)."""

    def __init__(
        self,
        perturbations: Weyl_Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        nuisance_params: dict,
        Jhat_params: dict,
        include_rsd: bool = False,
    ):
        # Call parent init but force per-bin bias model (Weyl only uses per-bin)
        super().__init__(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            galaxy_bias_model="per_bin",
            nuisance_params=nuisance_params,
            include_rsd=include_rsd,
        )

        # Weyl-specific fields
        self.Jhat_params = Jhat_params

        # Defines z_ini
        self.z_ini = self.perturbations.z_ini  # We can directly access z_ini from the Weyl_Perturbations instance, which is set at initialization of that class.

        # Get sigma8 at z_ini
        self.sigma8_ini = self.perturbations.sigma8_zini()

        # Override bias_array to use bhat_binN naming (bhat = b(z)*sigma8(z) in your scheme)
        bias_vals = np.asarray(
            [nuisance_params["bhat_bin%d" % bin] for bin in range(self.n_z_bins)]
        )
        # pad to match parent's expected length (parent used padding too)
        self.bias_array = np.pad(bias_vals, (0, self.z.shape[0] - self.n_z_bins))

        # Build Jhat_array from provided Jhat_params
        jhat_vals = np.asarray(
            [Jhat_params["Jhat_bin%d" % bin] for bin in range(self.n_z_bins)]
        )
        self.Jhat_array = np.pad(jhat_vals, (0, self.z.shape[0] - self.n_z_bins))

    def growth_since_zini(self, z) -> np.ndarray:
        """New function to account for the growth (in GR) since z_ini"""
        growth = (
            self.perturbations.growth_factor(z, self.perturbations.k[:1])[:, 0]
            / self.perturbations.growth_factor(
                np.array([self.z_ini]), self.perturbations.k[:1]
            )[0, 0]
        )
        return growth

    def get_window_positions(self, z) -> np.ndarray:
        """Weyl-modified positions window: multiplies by Jhat and by bhat; removes Omega_m^{-1}(z) factor;
        divides by sigma8_ini^2."""
        Omega_m = self.background.Omega_m(z)

        def per_bin_case():
            window = (
                self.bias_array[: self.n_z_bins, None]
                * self.Jhat_array[: self.n_z_bins, None]
                * self.dndz_shifted
                * self.perturbations.background.hubble_parameter(z)
                / (c_0 * Omega_m)
                / self.sigma8_ini**2
            )
            return window

        return per_bin_case()

    def get_window_rsd(self, ells, H, f, chi) -> np.ndarray:
        """Weyl GC RSD window: multiply by growth_factor (normalized to z_ini) once."""
        # S_i(z) = H(z) f(z) n_i(z) / c
        growth_factor = self.growth_since_zini(self.z)
        S = (H[None, :] * f[None, :] / c_0) * self.dndz_shifted * growth_factor**2
        return get_photo_rsd(ells, chi, S)

    def get_window_magnification(self, z):
        """Override magnification window: include growth factor squared (Weyl-specific)."""
        # Weyl project: added growth factor normalized to its value at z_ini
        growth_factor = self.growth_since_zini(z)

        Omega_m0 = self.background.Omega_m(0.0)
        factor = (
            3
            / 2
            * growth_factor**2
            * (self.background.H0 / c_0) ** 2
            * Omega_m0
            * (1 + z)
            * self.background.comoving_distance(z)
        )
        efficiency = self.get_magnification_efficiency(z)  # inherited from parent
        return (
            np.einsum("ij, j->ij", efficiency, factor)
            * np.array(self.magnification_bias)[:, None]
        )

    # get_magnification_efficiency and get_window are inherited unchanged from PositionsTracer

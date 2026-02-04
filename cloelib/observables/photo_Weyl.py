"""
Module with two classes for each observable tracer type: shear and galaxy positions.

Both classes are compatible with the Tracer protocol.
"""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations
from cloelib.observables.photo import PositionsTracer

# General imports
import jax.numpy as np  # type: ignore


# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s

# NOTE WEYL PROJECT: No changes to ShearTracer class --> import from usual photo.py file


class PositionsTracer_Weyl_GC(PositionsTracer):
    """Positions tracer for galaxy clustering including Weyl potential (subclass of PositionsTracer)."""

    def __init__(
        self,
        perturbations: Perturbations,  # Note: We should require this to be an instance of Weyl_perturbations
        dndz: np.ndarray,
        z: np.ndarray,
        nuisance_params: dict,
    ):
        # Reuse parent initialization but force per-bin bias model
        super().__init__(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            galaxy_bias_model="per_bin",
            nuisance_params=nuisance_params,
        )

        # Defines z_ini
        if hasattr(
            self.perturbations, "z_ini"
        ):  # True if perturbations is an instance of Weyl_Perturbations)
            self.z_ini = self.perturbations.z_ini
        else:
            # Fallback: use the first entry of perturbations.z,
            # but ensure the array has exactly one element.
            # Note: This will now lead to an error if perturbations is an instance of CAMB_perturbations (perturbations.z now enforced to contain 0 in recent changes). It still works with CLASS_perturbations. But we should probably remove this soon and enforce the use of the Weyl_perturbations class.
            if len(self.perturbations.z) != 1:
                raise ValueError(
                    f"Cannot infer z_ini from perturbations.z: expected length 1, "
                    f"got length {len(self.perturbations.z)}. "
                    "Multi-z arrays would lead to inconsistent C_ell calculations."
                )
            self.z_ini = self.perturbations.z[0]

        # Calculate sigma8 at z_ini by calling sigma8 at redshift 0 and multiplying by growth factor at z_ini (and dividing through growth factor today in case it's not already normalized to 1).
        self.sigma8_ini = (
            self.perturbations.sigma8_0()
            * self.perturbations.growth_factor(
                np.array([self.z_ini]), self.perturbations.k[:1]
            )[0, 0]
            / self.perturbations.growth_factor(np.array([0]), self.perturbations.k[:1])[
                0, 0
            ]
        )

        # Override bias_array to use bhat_binN naming (bhat = b(z)*sigma8(z))
        bias_vals = np.asarray(
            [
                nuisance_params.get("bhat_bin%d" % bin, 1.0)
                for bin in range(self.n_z_bins)
            ]
        )
        self.bias_array = np.pad(bias_vals, (0, self.z.shape[0] - self.n_z_bins))

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

    def get_window_magnification(self, z):
        """Weyl GC magnification: multiply by growth_factor (normalized to z_ini) once."""
        growth_factor = (
            self.perturbations.growth_factor(z, self.perturbations.k[:1])[:, 0]
            / self.perturbations.growth_factor(
                np.array([self.z_ini]), self.perturbations.k[:1]
            )[0, 0]
        )

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
        perturbations: Perturbations,
        dndz: np.ndarray,
        z: np.ndarray,
        nuisance_params: dict,
        Jhat_params: dict,
    ):
        # Call parent init but force per-bin bias model (Weyl only uses per-bin)
        super().__init__(
            perturbations=perturbations,
            dndz=dndz,
            z=z,
            galaxy_bias_model="per_bin",
            nuisance_params=nuisance_params,
        )

        # Weyl-specific fields
        self.Jhat_params = Jhat_params

        # Defines z_ini
        if hasattr(
            self.perturbations, "z_ini"
        ):  # True if perturbations is an instance of Weyl_Perturbations)
            self.z_ini = self.perturbations.z_ini
        else:
            # Fallback: use the first entry of perturbations.z,
            # but ensure the array has exactly one element.
            if len(self.perturbations.z) != 1:
                raise ValueError(
                    f"Cannot infer z_ini from perturbations.z: expected length 1, "
                    f"got length {len(self.perturbations.z)}. "
                    "Multi-z arrays would lead to inconsistent C_ell calculations."
                )
            self.z_ini = self.perturbations.z[0]

        # Calculate sigma8 at z_ini by calling sigma8 at redshift 0 and multiplying by growth factor at z_ini (and dividing through growth factor today in case it's not already normalized to 1).
        self.sigma8_ini = (
            self.perturbations.sigma8_0()
            * self.perturbations.growth_factor(
                np.array([self.z_ini]), self.perturbations.k[:1]
            )[0, 0]
            / self.perturbations.growth_factor(np.array([0]), self.perturbations.k[:1])[
                0, 0
            ]
        )

        # Override bias_array to use bhat_binN naming (bhat = b(z)*sigma8(z) in your scheme)
        bias_vals = np.asarray(
            [
                nuisance_params.get("bhat_bin%d" % bin, 1.0)
                for bin in range(self.n_z_bins)
            ]
        )
        # pad to match parent's expected length (parent used padding too)
        self.bias_array = np.pad(bias_vals, (0, self.z.shape[0] - self.n_z_bins))

        # Build Jhat_array from provided Jhat_params
        jhat_vals = np.asarray(
            [Jhat_params.get("Jhat_bin%d" % bin, 1.0) for bin in range(self.n_z_bins)]
        )
        self.Jhat_array = np.pad(jhat_vals, (0, self.z.shape[0] - self.n_z_bins))

    def get_window_positions(self, z) -> np.ndarray:
        """Weyl-modified positions window: multiplies by Jhat and by bhat; removes Omega_m^{-1}(z) factor;
        divides by sigma8_ini^2."""
        # compute Omega_m(z)
        Omega_m0 = self.background.Omega_m(0.0)
        Omega_m = (
            Omega_m0
            * (1 + z) ** 3
            * (self.background.H0 / self.background.hubble_parameter(z))
        )

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

    def get_window_magnification(self, z):
        """Override magnification window: include growth factor squared (Weyl-specific)."""
        # Weyl project: added growth factor normalized to its value at z_ini
        growth_factor = (
            self.perturbations.growth_factor(z, self.perturbations.k[:1])[:, 0]
            / self.perturbations.growth_factor(
                np.array([self.z_ini]), self.perturbations.k[:1]
            )[0, 0]
        )

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

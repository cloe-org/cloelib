"""Unit tests for the BNTMatrixCalculator class in cloelib.auxiliary.bnt (unittest style, pytest compatible)."""

import unittest
import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_allclose
from cloelib.auxiliary.bnt import BNTMatrixCalculator
from cloelib.cosmology.camb_cosmology import CAMBBackground

# --- Shared cosmology setup ---
fid_params = {
    "H0": 70.0,
    "Omega_cdm0": 0.25,
    "Omega_b0": 0.05,
    "Omega_k0": 0.0,
    "w0": -1.0,
    "wa": 0.0,
    "ns": 0.96,
    "As": 2.1e-9,
    "mnu": 0.06,
    "N_mnu": 3,
    "N_ur": 0.046,
    "gamma_MG": 0.55,
}
background = CAMBBackground(
    **{
        k: fid_params[k]
        for k in [
            "H0",
            "Omega_cdm0",
            "Omega_b0",
            "Omega_k0",
            "w0",
            "wa",
            "ns",
            "As",
            "mnu",
            "gamma_MG",
            "N_mnu",
            "N_ur",
        ]
    }
)


class TestBNTMatrixCalculator(unittest.TestCase):
    def test_bnt_matrix_correct_to_4dp(self):
        """Matrix matches expected values to 4 decimal places."""
        z = np.linspace(0.1, 2.0, 200)
        centers = [0.4, 0.8, 1.2]
        widths = [0.1, 0.1, 0.1]
        dndz_list = []
        for c, w in zip(centers, widths):
            nz = np.exp(-0.5 * ((z - c) / w) ** 2)
            nz /= np.trapz(nz, z)
            dndz_list.append(nz)

        bnt = BNTMatrixCalculator(dndz_list=dndz_list, z=z, background=background)
        BNT_matrix = bnt.get_bnt_matrix()

        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.2969388, -1.2969388, 1.0],
            ]
        )
        assert_allclose(BNT_matrix, expected, rtol=0, atol=1e-4)

    def test_bnt_matrix_correct_to_4dp_jax(self):
        """Matrix matches expected values to 4 decimal places when inputs are JAX arrays."""
        from numpy.testing import assert_allclose

        z = np.linspace(0.1, 2.0, 200)
        centers = [0.4, 0.8, 1.2]
        widths = [0.1, 0.1, 0.1]

        dndz_list = []
        for c, w in zip(centers, widths):
            nz = np.exp(-0.5 * ((z - c) / w) ** 2)
            nz /= np.trapz(nz, z)
            dndz_list.append(nz)

        # convert to jax before feeding to BNT
        z_jax = jnp.asarray(z)
        dndz_list_jax = [jnp.asarray(nz) for nz in dndz_list]

        bnt = BNTMatrixCalculator(
            dndz_list=dndz_list_jax,
            z=z_jax,
            background=background,
        )
        BNT_matrix = bnt.get_bnt_matrix()

        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.2969388, -1.2969388, 1.0],
            ]
        )

        assert_allclose(BNT_matrix, expected, rtol=0, atol=1e-4)

        assert isinstance(BNT_matrix, np.ndarray), "Output must be a NumPy array."

    def test_bnt_raises_when_dndz_shape_mismatch(self):
        """Raises ValueError if any dndz array has a shape different from z."""
        # z has length 100
        z = np.linspace(0.1, 2.0, 100)

        # Helper to build a normalized nz on a given z-grid
        def make_nz(z_grid, center, width):
            nz = np.exp(-0.5 * ((z_grid - center) / width) ** 2)
            nz /= np.trapz(nz, z_grid)
            return nz

        # One dndz with matching shape
        dndz_good = make_nz(z, center=0.8, width=0.1)

        # One dndz with mismatched shape (different grid)
        z_short = z[::2]  # length 50
        dndz_bad = make_nz(z_short, center=0.8, width=0.1)

        dndz_list = [dndz_good, dndz_bad, dndz_good]

        with self.assertRaisesRegex(
            ValueError,
            r"Each dndz array must have shape",
        ):
            BNTMatrixCalculator(
                dndz_list=dndz_list,
                z=z,
                background=background,
            )

    def test_bnt_raises_when_2x2_system_singular(self):
        """
        Raises ValueError if the 2x2 linear system in BNT construction is singular.

        We construct a case where the first two tomographic bins have identical
        n(z), so that for i=2 the matrix built from (A_{i-1}, B_{i-1}) and
        (A_{i-2}, B_{i-2}) has two identical rows and is therefore singular.
        """
        z = np.linspace(0.1, 2.0, 200)

        # Helper to build a normalised Gaussian n(z)
        def make_nz(z_grid, center, width):
            nz = np.exp(-0.5 * ((z_grid - center) / width) ** 2)
            nz /= np.trapz(nz, z_grid)
            return nz

        # First two bins identical → leads to singular 2×2 system for i=2
        nz_base = make_nz(z, center=0.8, width=0.1)
        d0 = nz_base.copy()
        d1 = nz_base.copy()

        # Third bin different, just to satisfy nbins >= 3
        d2 = make_nz(z, center=1.2, width=0.1)

        dndz_list = [d0, d1, d2]

        bnt = BNTMatrixCalculator(
            dndz_list=dndz_list,
            z=z,
            background=background,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"BNT matrix construction failed: non-invertible 2x2 system",
        ):
            bnt.get_bnt_matrix()

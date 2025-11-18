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
        import numpy as np
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

    def test_bnt_raises_when_fewer_than_three_bins(self):
        """Raises ValueError if fewer than 3 tomographic bins are provided."""
        z = np.linspace(0.1, 2.0, 200)

        def nz(c, w):
            g = np.exp(-0.5 * ((z - c) / w) ** 2)
            return g / np.trapz(g, z)

        d1 = nz(0.5, 0.1)
        d2 = nz(1.0, 0.1)

        with self.assertRaisesRegex(
            ValueError,
            r"BNTMatrixCalculator requires at least 3 tomographic bins to compute the matrix\.",
        ):
            BNTMatrixCalculator(
                dndz_list=[d1, d2],
                z=z,
                background=background,
            )

    def test_bnt_raises_when_zero_present_in_z(self):
        """Raises ValueError if any z element equals zero."""
        z = np.array([0.0, 0.5, 1.0], dtype=float)
        d = np.array([0.2, 0.3, 0.5], dtype=float)
        dndz_list = [d, d, d]

        with self.assertRaisesRegex(
            ValueError,
            r"One of the z array elements is equal to zero, breaking the BNT computation\.",
        ):
            BNTMatrixCalculator(
                dndz_list=dndz_list,
                z=z,
                background=background,
            )

    def test_bnt_raises_when_z_and_dndz_rank_mismatch(self):
        """Raises ValueError if z rank differs from dndz[i] rank."""
        z = np.linspace(0.1, 2.0, 80)
        g = np.exp(-0.5 * ((z - 0.8) / 0.1) ** 2)
        d1d = g / np.trapz(g, z)
        d2d = d1d[:, None]  # make it 2D to trigger rank mismatch

        with self.assertRaisesRegex(
            ValueError,
            r"The shape of the z array does not match the shape of the dndz array\.",
        ):
            BNTMatrixCalculator(
                dndz_list=[d2d, d2d, d2d],
                z=z,
                background=background,
            )

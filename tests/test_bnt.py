"""Unit tests for the BNT class in cloelib.auxiliary.bnt (unittest style, pytest compatible)."""

import unittest
import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_allclose
from cloelib.auxiliary.bnt import BNT
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


class TestBNT(unittest.TestCase):
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

        bnt = BNT(
            dndz_list=dndz_list, z=z, fid_parameters=fid_params, background=background
        )
        BNT_matrix = bnt.get_matrix()

        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.2969388, -1.2969388, 1.0],
            ]
        )
        assert_allclose(BNT_matrix, expected, rtol=0, atol=1e-4)

    def test_bnt_matrix_correct_to_4dp_jax(self):
        """Matrix matches expected values to 4 decimal places when using JAX arrays."""
        import jax.numpy as jnp
        import numpy as np
        from numpy.testing import assert_allclose

        def trapz_jax(y, x):
            """Minimal trapezoidal rule compatible with JAX, 1D arrays."""
            dx = x[1:] - x[:-1]
            return jnp.sum(0.5 * (y[1:] + y[:-1]) * dx)

        # Redshift grid as JAX array
        z = jnp.linspace(0.1, 2.0, 200, dtype=jnp.float64)

        # Build normalized n(z) distributions as JAX arrays
        centers = [0.4, 0.8, 1.2]
        widths = [0.1, 0.1, 0.1]
        dndz_list = []
        for c, w in zip(centers, widths):
            nz = jnp.exp(-0.5 * ((z - c) / w) ** 2)
            nz = nz / trapz_jax(nz, z)  # normalize with JAX-compatible trapz
            dndz_list.append(nz)

        # Compute BNT matrix using JAX backend
        bnt = BNT(
            dndz_list=dndz_list, z=z, fid_parameters=fid_params, background=background
        )
        BNT_matrix = bnt.get_matrix()

        # Expected result (NumPy for comparison)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.2969388, -1.2969388, 1.0],
            ],
            dtype=np.float64,
        )

        # Check that output is a JAX array
        assert isinstance(BNT_matrix, jnp.ndarray), (
            "BNT_matrix should be a jax.numpy.ndarray"
        )

        # Compare numerical values (convert JAX array to NumPy)
        assert_allclose(np.array(BNT_matrix), expected, rtol=0, atol=1e-4)

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
            r"BNT requires at least 3 tomographic bins to compute the matrix\.",
        ):
            BNT(
                dndz_list=[d1, d2],
                z=z,
                fid_parameters=fid_params,
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
            BNT(
                dndz_list=dndz_list,
                z=z,
                fid_parameters=fid_params,
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
            BNT(
                dndz_list=[d2d, d2d, d2d],
                z=z,
                fid_parameters=fid_params,
                background=background,
            )

    def test_bnt_raises_when_mixed_numpy_and_jax_arrays(self):
        """Raises TypeError when z and dndz_list come from different array backends."""

        # Case 1: z is NumPy, dndz_list is JAX
        z_np = np.linspace(0.1, 2.0, 100)
        dndz_jax = [jnp.linspace(0.1, 2.0, 100) for _ in range(3)]

        with self.assertRaisesRegex(TypeError, r"Mixed array backends detected"):
            BNT(
                dndz_list=dndz_jax,
                z=z_np,
                fid_parameters=fid_params,
                background=background,
            )

        # Case 2: z is JAX, dndz_list is NumPy
        z_jax = jnp.linspace(0.1, 2.0, 100)
        dndz_np = [np.linspace(0.1, 2.0, 100) for _ in range(3)]

        with self.assertRaisesRegex(TypeError, r"Mixed array backends detected"):
            BNT(
                dndz_list=dndz_np,
                z=z_jax,
                fid_parameters=fid_params,
                background=background,
            )

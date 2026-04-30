from cloelib.auxiliary.systematics import stretch_dndz_jax
import jax.numpy as jnp


def test_stretch_dndz_jax():
    z = jnp.linspace(0.0, 5.0, 2000)
    width = jnp.array([1.0, 0.5, 1.5])

    # Create Gaussian dN/dz per redshift bin
    mu = 2.5
    sigma = 0.2
    dndz_raw = jnp.exp(-0.5 * ((z - mu) / sigma) ** 2)
    dndz = jnp.stack([dndz_raw, dndz_raw, dndz_raw])

    stretched = stretch_dndz_jax(dndz, z, width)

    # Check that output shape matches
    assert stretched.shape == dndz.shape

    # Check that each row integrates to 1
    integral = (
        -0.5 * (stretched[:, 0] + stretched[:, -1]) + jnp.sum(stretched, axis=1)
    ) * (z[1] - z[0])
    assert jnp.allclose(integral, 1.0, rtol=1e-3), (
        "Output distributions are not normalized"
    )

    # Check identity of stretch 1
    input_norm = (-0.5 * (dndz[0, 0] + dndz[0, -1]) + jnp.sum(dndz[0])) * (z[1] - z[0])
    dndz_normalized = dndz[0] / input_norm
    assert jnp.allclose(stretched[0], dndz_normalized, atol=1e-4), (
        "Width 1.0 should return identity"
    )

    # Check mean preservation. Should not be changed by width stretches only
    def get_mean(nz, z_arr):
        return jnp.sum(z_arr * nz) * (z_arr[1] - z_arr[0])

    mean_orig = get_mean(dndz_normalized, z)
    mean_narrow = get_mean(stretched[1], z)  # width 0.5
    mean_wide = get_mean(stretched[2], z)  # width 1.5
    # Tolerance 1%, due to possible asymmetries at the edges
    assert jnp.allclose(mean_orig, mean_narrow, atol=1e-2), (
        "Mean shifted unexpectedly (narrow case)"
    )
    assert jnp.allclose(mean_orig, mean_wide, atol=1e-2), (
        "Mean shifted unexpectedly (wide case)"
    )

    # Check standard deviation
    def get_sigma(nz, z_arr, mean_val):
        var = jnp.sum(((z_arr - mean_val) ** 2) * nz) * (z_arr[1] - z_arr[0])
        return jnp.sqrt(var)

    sigma_orig = get_sigma(dndz_normalized, z, mean_orig)
    sigma_narrow = get_sigma(stretched[1], z, mean_narrow)
    sigma_wide = get_sigma(stretched[2], z, mean_wide)
    # Is the ratio approximately equal to the width parameter?
    assert jnp.allclose(sigma_wide / sigma_orig, 1.5, rtol=0.05), (
        "Stretching factor not reflected in sigma"
    )
    assert jnp.allclose(sigma_narrow / sigma_orig, 0.5, rtol=0.05), (
        "Compression factor not reflected in sigma"
    )

    # Optional: check zero-padding outside bounds
    assert jnp.all(stretched[:, 0] < 1e-4), "Left edge should be near zero"
    assert jnp.all(stretched[:, -1] < 1e-4), "Right edge should be near zero"

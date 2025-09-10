from cloelib.auxiliary.systematics import shift_dndz_jax
import jax.numpy as jnp


def test_shift_dndz_jax() -> None:
    z = jnp.linspace(0.0, 5.0, 500)
    dz = jnp.array([0.1, -0.2, 0.0])

    # Create Gaussian dN/dz per bin centered at different redshifts
    dndz = jnp.stack([jnp.exp(-0.5 * ((z - c) / 0.2) ** 2) for c in [1.0, 2.0, 3.0]])

    shifted = shift_dndz_jax(dndz, z, dz)

    # Check that output shape matches
    assert shifted.shape == dndz.shape

    # Check that each row integrates to 1
    integral = (-0.5 * (shifted[:, 0] + shifted[:, -1]) + jnp.sum(shifted, axis=1)) * (
        z[1] - z[0]
    )
    assert jnp.allclose(
        integral, 1.0, rtol=1e-3
    ), "Output distributions are not normalized"

    # Optional: check zero-padding outside bounds
    assert jnp.all(shifted[:, 0] < 1e-4), "Left edge should be near zero"
    assert jnp.all(shifted[:, -1] < 1e-4), "Right edge should be near zero"


import numpy as np
import jax.numpy as jnp

from cloelib.auxiliary.systematics import modes_dndz_jax


def test_modes_dndz_zero_amplitudes():
    z = jnp.linspace(0.0, 2.0, 101)
    dndz = jnp.array([
        jnp.exp(-0.5 * ((z - 0.5) / 0.15) ** 2),
        jnp.exp(-0.5 * ((z - 1.2) / 0.2) ** 2),
    ])

    modes = jnp.zeros((3, 2, 101))
    amplitudes = jnp.zeros(3)

    result = modes_dndz_jax(dndz, z, modes, amplitudes)

    expected = dndz / jnp.trapezoid(dndz, z, axis=1)[:, None]

    np.testing.assert_allclose(result, expected)


def test_modes_dndz_independent_bins():
    z = jnp.linspace(0.0, 2.0, 101)
    dndz = jnp.ones((2, 101))

    modes = jnp.zeros((2, 2, 101))
    modes = modes.at[0, 0].set(0.1 * jnp.sin(2 * jnp.pi * z / 2))
    modes = modes.at[1, 1].set(0.1 * jnp.sin(2 * jnp.pi * z / 2))

    amplitudes = jnp.array([1.0, 0.0])

    result = modes_dndz_jax(dndz, z, modes, amplitudes)

    assert not np.allclose(result[0], result[0, 0] * 0 + 1 / 2)
    np.testing.assert_allclose(result[1], jnp.ones(101) / 2)


def test_modes_dndz_clips_negative_values():
    z = jnp.linspace(0.0, 1.0, 101)
    dndz = jnp.ones((1, 101))

    modes = jnp.zeros((1, 1, 101))
    modes = modes.at[0, 0].set(-2.0 * z)

    amplitudes = jnp.ones(1)

    result = modes_dndz_jax(dndz, z, modes, amplitudes)

    assert jnp.all(result >= 0)
    assert jnp.any(result == 0)
    np.testing.assert_allclose(
        jnp.trapezoid(result, z, axis=1), 1.0
    )

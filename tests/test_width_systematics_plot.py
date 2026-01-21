import matplotlib.pyplot as plt
import jax.numpy as jnp
from cloelib.auxiliary.width_systematics import stretch_dndz_jax


def plot_stretch_dndz_jax():
    z = jnp.linspace(0.0, 5.0, 2000)
    width = jnp.array([1.0, 0.5, 1.5])

    # Create Gaussian dN/dz per redshift bin
    mu = 2.5
    sigma = 0.2
    dndz_raw = jnp.exp(-0.5 * ((z - mu) / sigma) ** 2)
    dndz = jnp.stack([dndz_raw, dndz_raw, dndz_raw])

    stretched = stretch_dndz_jax(dndz, z, width)

    plt.figure(figsize=(10, 6))

    # Reference for the mean
    plt.axvline(
        mu, color="gray", linestyle=":", alpha=0.5, label=f"Original Mean (z={mu})"
    )

    # Stretched n(z) (normalized)
    norm_factor = jnp.trapezoid(dndz_raw, z)
    plt.plot(
        z,
        dndz_raw / norm_factor,
        "k--",
        label="Input (Original)",
        alpha=0.3,
        linewidth=2,
    )

    colors = ["blue", "green", "red"]
    labels = ["Identity (width=1.0)", "Compressed (width=0.5)", "Stretched (width=1.5)"]

    for i in range(3):
        plt.plot(z, stretched[i], color=colors[i], label=labels[i], linewidth=2)

        new_mean = jnp.average(z, weights=stretched[i])
        print(f"Width {width[i]:.1f} -> New Mean: {new_mean:.4f} (Target: {mu})")

    plt.title("Stretched n(z)s")
    plt.xlabel("Redshift z")
    plt.ylabel("Normalized dN/dz")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, 3.5)

    plt.show()

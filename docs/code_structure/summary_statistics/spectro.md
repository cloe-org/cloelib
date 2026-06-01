# Spectroscopic Surveys Summary Statistics (Using SpectroPower)

## LegendreMultipoles

Compute multipoles $P_\ell(k)$ from 2D power spectrum $P(k, \mu)$.

**Location**: `cloelib/summary_statistics/legendre_multipoles.py`

**What it does**:

- Takes a `SpectroPower` object
- Integrates $P(k, \mu)$ over $\mu$ with Legendre polynomials
- Outputs $P_0(k)$, $P_2(k)$, $P_4(k)$, ...
- Allows to further compute two-point correlation functions, in cartesian and polar coordinates.

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.CometEFT_spectro import CometEFT_spectro
from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles

bg = CAMBBackground(H0=67.5, ...)
spectro = CometEFT_spectro(background=bg, z_pk=1.0)

# Compute multipoles
leg_multi = LegendreMultipoles(spectro_power=spectro)

k = np.logspace(-2, 0, 50)  # k in h/Mpc
multipoles = leg_multi.power_multipoles(
    k=k,
    ells=[0, 2, 4],  # Which multipoles
    z=1.0,
    bias=2.0,
)

P0 = multipoles[0]  # Monopole
P2 = multipoles[2]  # Quadrupole
P4 = multipoles[4]  # Hexadecapole

print(f"Monopole at k=0.1: {P0[20]:.2e}")
```

## BAOAlphas

Compute BAO distortion parameters $\alpha_\parallel$ and $\alpha_\perp$.

**Location**: `cloelib/summary_statistics/bao_alphas.py`

**What it does**: Compute alpha parameters for the BAO analysis

**Example**:

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.summary_statistics.bao_alphas import BaryonAcousticOscillations
import numpy as np

# True and fiducial cosmologies
bg = CAMBBackground(H0=67.5, ...)
bg_fid = CAMBBackground(H0=67.5, ...)  # Fiducial cosmology

zs = np.array([0.5, 1.0, 1.5])

bao = BaryonAcousticOscillations(
    background=bg,
    background_fiducial=bg_fid,
    redshifts=zs,
)

# Access alphas at each redshift
alphas_z05 = bao.alphas_dict[0.5]

alpha_parallel = alphas_z05['alpha_par']      # Along LOS
alpha_perpendicular = alphas_z05['alpha_perp'] # Transverse
alpha_iso = alphas_z05['alpha_iso']            # Isotropic dilation
alpha_AP = alphas_z05['alpha_AP']              # Alcock-Paczynski ratio
```


## Next Steps

The pipeline is now complete.
From here:

- [API Reference](../../api.md) – Full technical documentation
- [Contributing Guide](../../contributing.md) – General contribution guidelines
- [Playground Examples](https://github.com/cloe-org/playground) – Real usage examples
- [Back to Overview](../index.md) – Review the architecture

Or return to any component:

- [Background](../background.md)
- [Perturbations](../perturbations.md)
- [Observables](../observables.md)


## Next Steps

The pipeline is now complete.
From here:

- [API Reference](../../api.md) – Full technical documentation
- [Contributing Guide](../../contributing.md) – General contribution guidelines
- [Playground Examples](https://github.com/cloe-org/playground) – Real usage examples
- [Back to Overview](../index.md) – Review the architecture

Or return to any component:

- [Background](../background.md)
- [Perturbations](../perturbations.md)
- [Observables](../observables.md)
## Next Steps

The pipeline is now complete.
From here:

- [API Reference](../../api.md) – Full technical documentation
- [Contributing Guide](../../contributing.md) – General contribution guidelines
- [Playground Examples](https://github.com/cloe-org/playground) – Real usage examples
- [Back to Overview](../index.md) – Review the architecture

Or return to any component:

- [Background](../background.md)
- [Perturbations](../perturbations.md)
- [Observables](../observables.md)

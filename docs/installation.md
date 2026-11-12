# Installation

## Quick Start

1. **Set up your environment** – Create a fresh conda/mamba environment using the [cloe-org-environments](https://github.com/cloe-org/cloe-org-environments) repository. Then, clone `cloelib`:

```sh
git clone https://github.com/cloe-org/cloelib.git
cd cloelib
```

2. **Check out the latest release** (optional but recommended):

```sh
git checkout <latest-tag>  # Find tags in "Releases"
```

3. **Install cloelib** – Get the core library up and running, either from your clone:

```sh
pip install .
```

or directly from [PyPI](https://pypi.org/project/cloelib/) (no clone needed):

```sh
pip install cloelib
```

4. **Add optional dependencies** – Enhance with external codes and tools:

```sh
pip install .[camb,classy,comet-emu,euclidemu2,pylevin,mpmath,tensorflow,tf-keras,cosmopower-jax,baccoemu,hiclassy,emantis,pbjcosmo,fastpt,react,mgclassy,FlamingoBaryonResponseEmulator]
```

> **Note:** Some shells require quoting the argument when brackets are present:
>
> ```sh
> pip install ".[camb,classy,comet-emu,euclidemu2,pylevin,mpmath,tensorflow,tf-keras,cosmopower-jax,baccoemu,hiclassy,emantis,pbjcosmo,fastpt,react,mgclassy,FlamingoBaryonResponseEmulator]"
> ```

5. **Install backends not available on PyPI** – HMcode2020Emu, MGEmu (needed by the `react` module) and mochi_class are only distributed from GitHub, so they cannot be declared as extras of a package published on PyPI. Install the ones you need directly:

```sh
pip install "HMcode2020Emu @ git+https://github.com/MariaTsedrik/HMcode2020Emu.git"
pip install "MGemu @ git+https://github.com/nebblu/MGEmus.git"
pip install "mochi_classy @ git+https://github.com/mcataneo/mochi_class_public.git@mochi_classy"
```

The installation is now complete.

## Optional Dependencies

Several optional dependencies enhance **cloelib** capabilities:

- **`pyinstrument`** – Time profiling for performance optimization
- **`pylevin`**, **`mpmath`** – Required for specific observational probes (e.g., COSEBIs)
- **`tensorflow`** – Needed for certain emulator backends (e.g., `HMCode2020emu`)
- **`react`** – Installs the TensorFlow support needed by the ReACT modified-gravity boost module; [`MGEmu`](https://github.com/nebblu/MGEmus.git) itself must be installed from GitHub (see step 5)
- **`baccoemu`** – Required for BACCOemu nonlinear and baryonic perturbations
- **`FlamingoBaryonResponseEmulator`** – Required for FLAMINGO baryonic suppression

These are not included in the default installation but can be added as shown above as other external cosmological codes.

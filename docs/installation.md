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

3. **Install cloelib** – Get the core library up and running:

```sh
pip install .
```

4. **Add optional dependencies** – Enhance with external codes and tools:

```sh
pip install .[camb,classy,hmcode2020emu,comet-emu,euclidemu2,pylevin,mpmath,tensorflow,pyinstrument,baccoemu,FlamingoBaryonResponseEmulator]
```

> **Note:** Some shells require quoting the argument when brackets are present:
>
> ```sh
> pip install ".[camb,classy,hmcode2020emu,comet-emu,euclidemu2,pylevin,mpmath,tensorflow,pyinstrument,baccoemu,FlamingoBaryonResponseEmulator]"
> ```

The installation is now complete.

## Optional Dependencies

Several optional dependencies enhance **cloelib** capabilities:

- **`pyinstrument`** – Time profiling for performance optimization
- **`pylevin`**, **`mpmath`** – Required for specific observational probes (e.g., COSEBIs)
- **`tensorflow`** – Needed for certain emulator backends (e.g., `HMCode2020emu`)
- **`baccoemu`** – Required for BACCOemu nonlinear and baryonic perturbations
- **`FlamingoBaryonResponseEmulator`** – Required for FLAMINGO baryonic suppression

These are not included in the default installation but can be added as shown above as other external cosmological codes.

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
pip install .[camb,classy,hmcode2020emu,comet-emu,pylevin,mpmath,tensorflow,pyinstrument]
```

> **💡 Pro Tip:** Some shells struggle with brackets. Try quotes if needed:
>
> ```sh
> pip install ".[camb,classy,hmcode2020emu,comet-emu,pylevin,mpmath,tensorflow,pyinstrument]"
> ```

You're all set! 🎉 Ready to compute cosmological observables.

## Optional Dependencies

Several optional dependencies enhance **cloelib** capabilities:

- **`pyinstrument`** – Time profiling for performance optimization
- **`pylevin`**, **`mpmath`** – Required for specific observational probes (e.g., COSEBIs)
- **`tensorflow`** – Needed for certain emulator backends (e.g., `HMCode2020emu`)

These are not included in the default installation but can be added as shown above as other external cosmological codes.

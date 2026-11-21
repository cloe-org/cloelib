"""Spectroscopic-probe interfaces: the `SpectroPower` protocol and its
PT/emulator backends (PBJ, PyBird, Comet).

`spectro.spectro` holds the `SpectroPower` protocol itself;
`spectro.PBJ_spectro`, `spectro.PyBird_spectro`, `spectro.CometEFT_spectro`
and `spectro.CometVDG_spectro` each hold one backend implementing it. Grouped
together here for file-tree organization; moved from the old flat
`cloelib/observables/PBJ_spectro.py`/`CometEFT_spectro.py`/`CometVDG_spectro.py`
(imports updated at every call site - no back-compat shims left at the old
paths).
"""

from cloelib.observables.spectro import spectro
from cloelib.observables.spectro.spectro import SpectroPower

__all__ = ["SpectroPower", "spectro"]

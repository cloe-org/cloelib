"""CMB tracers: CMB lensing convergence and thermal Sunyaev-Zel'dovich.

`CMBLensingTracer` lives in `cmb.lensing`, and `tSZTracer` (Compton-y) lives
in `cmb.tsz`. Both implement the tracer interface consumed by
`AngularTwoPoint` and are re-exported here for concise imports.
"""

from cloelib.observables.cmb import lensing, tsz
from cloelib.observables.cmb.lensing import CMBLensingTracer
from cloelib.observables.cmb.tsz import tSZTracer

__all__ = [
    "CMBLensingTracer",
    "tSZTracer",
    "lensing",
    "tsz",
]

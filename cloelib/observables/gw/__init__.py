"""Gravitational-wave number-count and weak-lensing tracers.

`GWNumberCountsTracer` lives in `gw.number_counts`, and
`GWWeakLensingTracer` lives in `gw.weak_lensing`. Both implement the tracer
interface consumed by `AngularTwoPoint` and are re-exported here for concise
imports.
"""

from cloelib.observables.gw import number_counts, weak_lensing
from cloelib.observables.gw.number_counts import GWNumberCountsTracer
from cloelib.observables.gw.weak_lensing import GWWeakLensingTracer

__all__ = [
    "GWNumberCountsTracer",
    "GWWeakLensingTracer",
    "number_counts",
    "weak_lensing",
]

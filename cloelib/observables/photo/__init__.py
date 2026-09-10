"""Photometric probe tracers: cosmic shear and angular galaxy clustering.

`ShearTracer` and everything related to its intrinsic-alignment models
(`LensingContribution`, `NLAContribution`,
`TATTContribution`, `PBJTATTLoopComputer`) live together in `photo.shear`.
`PositionsTracer` and its Contributions
(`GalaxyBiasContribution`, `MagnificationContribution`) live together in
`photo.positions`. The generic, tracer-agnostic pieces both of those (and
`AngularTwoPoint`) build on - the `Tracer` protocol, the `Contribution`
protocol/`IntrinsicAlignmentContribution` base, and the generalized
per-contribution-pair spectrum engine - live here too, in `photo.tracer`,
`photo.contributions`, `photo.spectrum_engine` respectively; moved from the
old flat `cloelib/observables/tracer.py`/`contributions.py`/
`spectrum_engine.py` (imports updated at every call site - no back-compat
shims left at the old paths).

`ShearTracer` and `PositionsTracer` are re-exported here too, for
`from cloelib.observables.photo import ShearTracer, PositionsTracer`.
"""

from cloelib.observables.photo import (
    contributions,
    positions,
    shear,
    spectrum_engine,
    tracer,
)
from cloelib.observables.photo.positions import PositionsTracer
from cloelib.observables.photo.shear import ShearTracer

__all__ = [
    "ShearTracer",
    "PositionsTracer",
    "shear",
    "positions",
    "tracer",
    "contributions",
    "spectrum_engine",
]

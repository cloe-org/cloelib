from typing import TYPE_CHECKING, Protocol, TypeVar, Union

import numpy as np  # type: ignore
import jax.numpy as jnp  # type: ignore

if TYPE_CHECKING:
    from cloelib.observables.photo import PositionsTracer, ShearTracer

T = TypeVar("T", bound=Union[jnp.ndarray, np.ndarray])


class Contribution(Protocol):
    """A single additive term of a tracer's window function."""

    def compute_kernel(self, z: T) -> T:
        """Return this contribution's kernel, shape (n_bins, len(z))."""
        ...


class LensingContribution:
    """Weak-lensing shear kernel term of `ShearTracer.get_window()`."""

    def __init__(self, tracer: "ShearTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z: T) -> T:
        return self._tracer.get_window_lensing(z)


class IntrinsicAlignmentContribution:
    """Intrinsic-alignment kernel term of `ShearTracer.get_window()`.

    Currently always the NLA model implemented by
    `ShearTracer.get_window_IA`. This is the seam a future TATT contribution
    would replace - swapping the model would mean a tracer holding a
    different `Contribution` here, not a change to `get_window()` or to
    `AngularTwoPoint`.
    """

    def __init__(self, tracer: "ShearTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z: T) -> T:
        return self._tracer.get_window_IA(z)


class GalaxyBiasContribution:
    """Galaxy-bias-weighted positions kernel term of `PositionsTracer.get_window()`.

    Currently one of the three linear-bias models selected by
    `galaxy_bias_model` (`ShearTracer.get_window_positions`). This is the
    seam a future non-linear galaxy bias contribution would occupy instead.
    """

    def __init__(self, tracer: "PositionsTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z: T) -> T:
        return self._tracer.get_window_positions(z)


class MagnificationContribution:
    """Magnification-bias kernel term of `PositionsTracer.get_window()`."""

    def __init__(self, tracer: "PositionsTracer") -> None:
        self._tracer = tracer

    def compute_kernel(self, z: T) -> T:
        return self._tracer.get_window_magnification(z)

"""
Tests for `AngularTwoPoint.get_pseudo_Cl`, i.e. the convolution of the
theory Cls with the mixing matrices.

Conventions used here (euclidlib internal format):
    - POS-POS: one matrix M, ``C_out = M @ C``.
    - POS-SHE: one matrix M applied to each of the E and B components.
    - SHE-SHE: ``array[0] = M_EE``, ``array[1] = M_BB``, ``array[2] = M_EB``,
          EE_out = M_EE @ EE + M_BB @ BB
          BB_out = M_BB @ EE + M_EE @ BB
          EB_out = M_EB @ EB
          BE_out = M_EB @ BE
"""

import numpy as np
import pytest
from cosmolib.data import AngularPowerSpectrum

from cloelib.observables.photo import PositionsTracer, ShearTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

N_BIN = 2
ELLMAX = 12
N_ELL_OUT = 4

TRACER_KEYS = {
    (PositionsTracer, PositionsTracer): ("POS", "POS"),
    (PositionsTracer, ShearTracer): ("POS", "SHE"),
    (ShearTracer, PositionsTracer): ("POS", "SHE"),
    (ShearTracer, ShearTracer): ("SHE", "SHE"),
}

# Shape of each spectrum / mixing matrix, excluding the trailing ell axes.
CL_COMPONENTS = {("POS", "POS"): (), ("POS", "SHE"): (2,), ("SHE", "SHE"): (2, 2)}
MM_COMPONENTS = {("POS", "POS"): (), ("POS", "SHE"): (), ("SHE", "SHE"): (3,)}


def _stub_tracer(cls):
    """Tracer of the right type for dispatch, carrying only `n_z_bins`."""
    tracer = object.__new__(cls)
    tracer.n_z_bins = N_BIN
    return tracer


def _bin_pairs(key_type):
    """Bin pairs `get_pseudo_Cl` reads: all of them for POS-SHE, i <= j otherwise."""
    return [
        (i, j)
        for i in range(1, N_BIN + 1)
        for j in range(1, N_BIN + 1)
        if key_type == ("POS", "SHE") or i <= j
    ]


def _mixing_matrix(array):
    ell_out = np.arange(N_ELL_OUT) * 3.0 + 2.0
    return AngularPowerSpectrum(
        array=array,
        ell=ell_out,
        lower=ell_out - 1.0,
        upper=ell_out + 2.0,
    )


def _run_pseudo_cl(tracer_types, cls, mms, monkeypatch):
    """Call `get_pseudo_Cl` with `get_Cl` stubbed to return `cls`."""
    a2p = AngularTwoPoint(*(_stub_tracer(t) for t in tracer_types))

    def fake_get_Cl(ells, nl, ks):
        # get_pseudo_Cl must ask for every ell the mixing matrix acts on.
        np.testing.assert_array_equal(ells, np.arange(ELLMAX))
        return {
            key: AngularPowerSpectrum(array=arr, ell=np.arange(ELLMAX))
            for key, arr in cls.items()
        }

    monkeypatch.setattr(a2p, "get_Cl", fake_get_Cl)
    mixing = {key: _mixing_matrix(arr) for key, arr in mms.items()}
    return a2p.get_pseudo_Cl(nl=None, ks=None, mixing_matrix=mixing), mixing


def _expected(key_type, C, M):
    """Independent reference for the mixing, written with einsum."""
    if key_type == ("SHE", "SHE"):
        M_EE, M_BB, M_EB = M
        out = np.empty((2, 2, N_ELL_OUT))
        out[0, 0] = M_EE @ C[0, 0] + M_BB @ C[1, 1]
        out[1, 1] = M_BB @ C[0, 0] + M_EE @ C[1, 1]
        out[0, 1] = M_EB @ C[0, 1]
        out[1, 0] = M_EB @ C[1, 0]
        return out
    return np.einsum("ol,...l->...o", M, C)


def test_she_she_simple_example(monkeypatch):
    """Hand-checkable SHE-SHE case with scaled-identity mixing matrices.

    With M_EE = a*I, M_BB = b*I, M_EB = c*I and constant input spectra
    EE, BB, EB, BE the output is known in closed form. Every input
    component has a distinct value, so using the wrong one (as in gh-306)
    changes the result.
    """
    a, b, c = 0.7, 0.2, 0.5
    EE, BB, EB, BE = 1.0, 3.0, 5.0, 11.0

    identity = np.eye(N_ELL_OUT, ELLMAX)
    mm = np.stack([a * identity, b * identity, c * identity])
    cl = np.empty((2, 2, ELLMAX))
    cl[0, 0], cl[1, 1], cl[0, 1], cl[1, 0] = EE, BB, EB, BE

    key = ("SHE", "SHE", 1, 1)
    keys = [("SHE", "SHE", i, j) for i, j in _bin_pairs(("SHE", "SHE"))]
    out, _ = _run_pseudo_cl(
        (ShearTracer, ShearTracer),
        {k: cl for k in keys},
        {k: mm for k in keys},
        monkeypatch,
    )

    result = out[key].array
    np.testing.assert_allclose(result[0, 0], a * EE + b * BB)
    np.testing.assert_allclose(result[1, 1], b * EE + a * BB)
    np.testing.assert_allclose(result[0, 1], c * EB)
    np.testing.assert_allclose(result[1, 0], c * BE)


@pytest.mark.parametrize(
    "tracer_types",
    list(TRACER_KEYS),
    ids=["POS-POS", "POS-SHE", "SHE-POS", "SHE-SHE"],
)
def test_pseudo_cl_matches_reference(tracer_types, monkeypatch):
    """Random spectra and mixing matrices, for every supported tracer pair
    and every bin pair, against the reference in `_expected`."""
    rng = np.random.default_rng(309)
    key_type = TRACER_KEYS[tracer_types]
    keys = [key_type + pair for pair in _bin_pairs(key_type)]

    cls = {k: rng.normal(size=CL_COMPONENTS[key_type] + (ELLMAX,)) for k in keys}
    mms = {
        k: rng.normal(size=MM_COMPONENTS[key_type] + (N_ELL_OUT, ELLMAX)) for k in keys
    }

    out, mixing = _run_pseudo_cl(tracer_types, cls, mms, monkeypatch)

    assert set(out) == set(keys)
    for k in keys:
        np.testing.assert_allclose(
            out[k].array, _expected(key_type, cls[k], mms[k]), rtol=1e-12
        )
        # Binning metadata is carried over from the mixing matrix (gh-306).
        np.testing.assert_array_equal(out[k].ell, mixing[k].ell)
        np.testing.assert_array_equal(out[k].lower, mixing[k].lower)
        np.testing.assert_array_equal(out[k].upper, mixing[k].upper)


def test_pseudo_cl_unsupported_tracers():
    a2p = AngularTwoPoint(object(), object())
    with pytest.raises(ValueError, match="Unsupported tracer pair"):
        a2p.get_pseudo_Cl(nl=None, ks=None, mixing_matrix={})

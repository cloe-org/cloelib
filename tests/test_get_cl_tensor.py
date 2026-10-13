"""
Tests for `AngularTwoPoint.get_Cl_tensor`, the packaging-free counterpart
to `get_Cl` (see its docstring in `angular_two_point.py`, and the
investigation this came out of, `playground/tutorials/observables/
photo_autodiff.ipynb`).

`get_Cl`'s packaging step wraps the result in `cosmolib.data.photo.
AngularPowerSpectrum`; that dataclass's `__post_init__` used to
unconditionally do `np.asarray(self.array, dtype=float)` - a plain NumPy
cast that severed any `jax.grad` trace passing through it. `cosmolib`'s
`26-fix-jax-clash-with-cloelib-photo-classes` fix made that cast
JAX-aware, so `get_Cl` itself is differentiable now too - `get_Cl_tensor`
remains worth using on its own merits (skips building the packaged `dict`
and `AngularPowerSpectrum` objects), not as a differentiability
workaround. These tests use `cloelib.cosmology.jax_cosmology`
(`JAXBackground`, `JAXNonLinearPerturbations`) specifically, since it's
the one cosmology backend that's pure JAX end-to-end - CAMB/
HMcode2020Emu/etc. call non-JAX external codes internally regardless of
this method.
"""

import importlib.util

import jax
import jax.numpy as jnp
import pytest

from cloelib.cosmology.jax_cosmology import JAXBackground, JAXNonLinearPerturbations
from cloelib.observables.photo import ShearTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

_FASTPT_INSTALLED = importlib.util.find_spec("fastpt") is not None
if _FASTPT_INSTALLED:
    from cloelib.observables.photo.shear import PBJTATTLoopComputer


@pytest.fixture(scope="module")
def grids():
    return {
        "z_tracer": jnp.linspace(0.2, 2.0, 15),
        "z_grid": jnp.linspace(0.01, 3.0, 50),
        "ks": jnp.logspace(-4, 1, 50),
        "ells": jnp.logspace(1.0, jnp.log10(200), 5),
    }


def _build_perturbations(z_grid, ks, H0=67.7):
    background = JAXBackground(
        H0=H0,
        Omega_b0=0.05,
        Omega_cdm0=0.25,
        Omega_k0=0.0,
        As=2e-9,
        ns=0.96,
        mnu=0.06,
        w0=-1.0,
        wa=0.0,
        gamma_MG=0.0,
        N_mnu=1,
    )
    perturbations = JAXNonLinearPerturbations(background=background)
    # JAXNonLinearPerturbations evaluates P(k,z) on the fly - it has no
    # natural pre-tabulated grid, so `.z`/`.k` (the informal contract
    # AngularTwoPoint.get_Cl relies on) are set explicitly here.
    perturbations.z = z_grid
    perturbations.k = ks
    return perturbations


def _nuisance_shear(**extra):
    return {
        "multiplicative_bias_1": 0.0,
        "dz_shear_1": 0.0,
        "width_shear_1": 1.0,
        "AIA": 1.0,
        "CIA": 0.0134,
        "EtaIA": -0.41,
        **extra,
    }


def test_get_cl_tensor_matches_get_cl_packaged_values(grids):
    """`get_Cl_tensor`'s raw tensor and `get_Cl`'s packaged EE block must
    carry the exact same numbers - `get_Cl_tensor` only skips packaging,
    it must not change any physics.
    """
    perturbations = _build_perturbations(grids["z_grid"], grids["ks"])
    dndz = jnp.ones((1, len(grids["z_tracer"])))
    dndz = dndz / jnp.trapezoid(dndz, grids["z_tracer"], axis=1)[:, None]
    tracer = ShearTracer(
        perturbations=perturbations,
        dndz=dndz,
        z=grids["z_tracer"],
        nuisance_params=_nuisance_shear(),
        ia_model="NLA",
    )
    two_point = AngularTwoPoint(tracer, tracer)

    cl_tensor = two_point.get_Cl_tensor(grids["ells"], 0, grids["ks"])
    cl_packaged = two_point.get_Cl(grids["ells"], 0, grids["ks"])

    assert cl_tensor.shape == (len(grids["ells"]), 1, 1)
    ee_packaged = cl_packaged[("SHE", "SHE", 1, 1)].array[0, 0]
    assert jnp.allclose(ee_packaged, cl_tensor[:, 0, 0])


def test_get_cl_is_now_directly_differentiable(grids):
    """`get_Cl` itself must differentiate cleanly now (no `get_Cl_tensor`
    workaround needed), now that `cosmolib`'s
    `26-fix-jax-clash-with-cloelib-photo-classes` fix stopped
    `AngularPowerSpectrum.__post_init__` from casting a JAX trace to plain
    NumPy. Its gradient must also match `get_Cl_tensor`'s exactly - same
    physics, only the packaging differs.
    """

    def loss_packaged(H0):
        perturbations = _build_perturbations(grids["z_grid"], grids["ks"], H0=H0)
        dndz = jnp.ones((1, len(grids["z_tracer"])))
        dndz = dndz / jnp.trapezoid(dndz, grids["z_tracer"], axis=1)[:, None]
        tracer = ShearTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=grids["z_tracer"],
            nuisance_params=_nuisance_shear(),
            ia_model="NLA",
        )
        cl = AngularTwoPoint(tracer, tracer).get_Cl(grids["ells"], 0, grids["ks"])
        return jnp.sum(cl[("SHE", "SHE", 1, 1)].array)

    def loss_tensor(H0):
        perturbations = _build_perturbations(grids["z_grid"], grids["ks"], H0=H0)
        dndz = jnp.ones((1, len(grids["z_tracer"])))
        dndz = dndz / jnp.trapezoid(dndz, grids["z_tracer"], axis=1)[:, None]
        tracer = ShearTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=grids["z_tracer"],
            nuisance_params=_nuisance_shear(),
            ia_model="NLA",
        )
        two_point = AngularTwoPoint(tracer, tracer)
        return jnp.sum(two_point.get_Cl_tensor(grids["ells"], 0, grids["ks"])[:, 0, 0])

    val_packaged, grad_packaged = jax.value_and_grad(loss_packaged)(67.7)
    val_tensor, grad_tensor = jax.value_and_grad(loss_tensor)(67.7)

    assert jnp.isfinite(grad_packaged) and grad_packaged != 0.0
    assert jnp.allclose(val_packaged, val_tensor)
    assert jnp.allclose(grad_packaged, grad_tensor)

    eps = 1e-4
    fd_H0 = (loss_packaged(67.7 + eps) - loss_packaged(67.7 - eps)) / (2 * eps)
    assert jnp.allclose(grad_packaged, fd_H0, rtol=1e-3)


def test_get_cl_tensor_differentiable_nla(grids):
    """`get_Cl_tensor` must differentiate through both a nuisance
    parameter (AIA) and a cosmological one (H0, via JAXBackground/
    JAXNonLinearPerturbations), cross-checked against finite differences.
    """

    def loss(AIA, H0):
        perturbations = _build_perturbations(grids["z_grid"], grids["ks"], H0=H0)
        dndz = jnp.ones((1, len(grids["z_tracer"])))
        dndz = dndz / jnp.trapezoid(dndz, grids["z_tracer"], axis=1)[:, None]
        tracer = ShearTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=grids["z_tracer"],
            nuisance_params=_nuisance_shear(AIA=AIA),
            ia_model="NLA",
        )
        two_point = AngularTwoPoint(tracer, tracer)
        return jnp.sum(two_point.get_Cl_tensor(grids["ells"], 0, grids["ks"]))

    grad_AIA = jax.grad(loss, argnums=0)(1.0, 67.7)
    grad_H0 = jax.grad(loss, argnums=1)(1.0, 67.7)
    assert jnp.isfinite(grad_AIA) and grad_AIA != 0.0
    assert jnp.isfinite(grad_H0) and grad_H0 != 0.0

    eps = 1e-4
    fd_H0 = (loss(1.0, 67.7 + eps) - loss(1.0, 67.7 - eps)) / (2 * eps)
    assert jnp.allclose(grad_H0, fd_H0, rtol=1e-3)


@pytest.mark.skipif(not _FASTPT_INSTALLED, reason="fast-pt not installed")
def test_get_cl_tensor_differentiable_tatt(grids):
    """Same check for TATT: differentiable w.r.t. its own amplitude
    parameters (the one-loop kernels are fixed multiplicative weights as
    far as those are concerned - see `PBJTATTLoopComputer`'s docstring for
    why it can't also differentiate through cosmological parameters that
    would affect the kernels themselves: FAST-PT is plain NumPy/SciPy).
    """
    perturbations = _build_perturbations(grids["z_grid"], grids["ks"])
    dndz = jnp.ones((1, len(grids["z_tracer"])))
    dndz = dndz / jnp.trapezoid(dndz, grids["z_tracer"], axis=1)[:, None]

    def loss(A2IA):
        tracer = ShearTracer(
            perturbations=perturbations,
            dndz=dndz,
            z=grids["z_tracer"],
            nuisance_params=_nuisance_shear(A2IA=A2IA, bTA=-0.83),
            ia_model="TATT",
            tatt_loop_computer=PBJTATTLoopComputer(perturbations),
        )
        two_point = AngularTwoPoint(tracer, tracer)
        return jnp.sum(two_point.get_Cl_tensor(grids["ells"], 0, grids["ks"]))

    val, grad_A2IA = jax.value_and_grad(loss)(0.40)
    assert jnp.isfinite(val) and jnp.isfinite(grad_A2IA)

    eps = 1e-4
    fd = (loss(0.40 + eps) - loss(0.40 - eps)) / (2 * eps)
    assert jnp.allclose(grad_A2IA, fd, rtol=1e-2)

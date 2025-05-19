import numpy as np
from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.cosmology.camb_cosmology import (
    CAMBBackground,
    CAMBLinearPerturbations,
    CAMBNonLinearPerturbations,
)

from cloelib.cosmology.class_cosmology import (
    CLASSBackground,
    CLASSLinearPerturbations,
    CLASSNonLinearPerturbations,
)
from cloelib.cosmology.jax_cosmology import (
    JAXBackground,
    JAXLinearPerturbations,
    JAXNonLinearPerturbations,
)

# from cloelib.cosmology.HMcode2020Emu_cosmology import (
#    HMemuLinearPerturbations,
#    HMemuNonLinearPerturbations,
# )


def _safe_ni_assert(assert_func, func, args, kwargs, reference, **kwargs_assert):
    if kwargs_assert is None:
        kwargs_assert = {}
    try:
        assert_func(func(*args, **kwargs), reference, **kwargs_assert)
    except Exception as e:
        if not isinstance(e, NotImplementedError):
            raise e


def assert_less_than(value_smaller, value_larger):
    assert value_smaller < value_larger


def test_cosmo():
    # Cosmology parameters
    print("# Cosmology parameters")
    _H0 = 67.7
    _h = _H0 / 100.0
    _omch2 = 0.12
    _ombh2 = 0.022
    _cosmo_pars = dict(
        H0=_H0,
        Omega_cdm0=_omch2 / _h**2,
        Omega_b0=_ombh2 / _h**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=0.96,
        mnu=0.1,
        As=2e-9,
        gamma_MG=0.0,
    )

    _z_test = np.zeros(1)
    _z_init = np.linspace(0.0, 2.0, 100)

    for (
        _bkg_list,
        (_LinearPerturbations, _lp_args),
        (_NonLinearPerturbations, _nlp_args),
    ) in (
        (
            (CAMBBackground,),
            (CAMBLinearPerturbations, (_z_init,)),
            (CAMBNonLinearPerturbations, (_z_init,)),
        ),
        (
            (CLASSBackground,),
            (CLASSLinearPerturbations, (_z_init,)),
            (CLASSNonLinearPerturbations, (_z_init, "halofit")),
        ),
        (
            (JAXBackground,),
            (JAXLinearPerturbations, (_z_init,)),
            (JAXNonLinearPerturbations, ()),
        ),
        # (
        #   (CAMBBackground, CLASSBackground, JAXBackground,),
        #   (HMemuLinearPerturbations, (_z_init,)),
        #   (HMemuNonLinearPerturbations, (_z_init,)),
        # ),
    ):
        for _Background in _bkg_list:
            # background

            background = _Background(**_cosmo_pars)
            _safe_ni_assert(
                assert_less_than,
                background.Omega_m_cb,
                (_z_test,),
                {},
                background.Omega_m(_z_test),
            )
            _safe_ni_assert(
                assert_allclose,
                background.Omega_m_cb,
                (_z_test,),
                {},
                _cosmo_pars["Omega_cdm0"] + _cosmo_pars["Omega_b0"],
                rtol=1e-07,
            )
            if hasattr(background, "rho_crit"):
                assert_allclose(background.rho_crit(0), 1.27203085e11, rtol=1e-07)
            _safe_ni_assert(
                assert_allclose,
                lambda: getattr(background, "rdrag"),
                (),
                {},
                147.50225,
                rtol=1e-7,
            )

            # linear
            perturbations = _LinearPerturbations(background, *_lp_args)
            _safe_ni_assert(
                assert_allclose,
                perturbations.matter_power_spectrum_cb,
                (0, 1),
                {},
                81.748209,
            )

            # non-linear
            perturbations_nl = _NonLinearPerturbations(background, *_nlp_args)
            _safe_ni_assert(
                assert_allclose,
                perturbations_nl.matter_power_spectrum_cb,
                (0, 1),
                {},
                747.017036,
            )

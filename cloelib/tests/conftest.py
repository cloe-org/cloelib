"""
Shared fixtures and configuration for cosmology tests.

This module provides common fixtures for testing cosmology backends,
reducing code duplication across individual test files.
"""

import pytest
import numpy as np


def pytest_addoption(parser):
    """Add command line option for backend selection."""
    parser.addoption(
        "--backend",
        action="store", 
        default="all",
        choices=["camb", "class", "jax", "all"],
        help="Select which cosmology backend to test (default: all)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "camb: mark test to run only with CAMB backend")
    config.addinivalue_line("markers", "class: mark test to run only with CLASS backend") 
    config.addinivalue_line("markers", "jax: mark test to run only with JAX backend")


def get_backend_implementations(selected_backend=None):
    """
    Dynamically discover available cosmology backend implementations.
    
    Parameters
    ----------
    selected_backend : str, optional
        Backend to filter for. Options: 'camb', 'class', 'jax', 'all' or None.
        If None or 'all', returns all available backends.
    
    Returns
    -------
    dict
        Dictionary mapping backend names to their implementation classes.
        Format: {backend_name: {
            'Background': BackgroundClass,
            'LinearPerturbations': LinearPerturbationsClass, 
            'NonLinearPerturbations': NonLinearPerturbationsClass
        }}
    """
    all_implementations = {}
    
    # Try to import CAMB backend
    try:
        from cloelib.cosmology.camb_cosmology import (
            CAMBBackground, CAMBLinearPerturbations, CAMBNonLinearPerturbations
        )
        all_implementations['CAMB'] = {
            'Background': CAMBBackground,
            'LinearPerturbations': CAMBLinearPerturbations,
            'NonLinearPerturbations': CAMBNonLinearPerturbations
        }
    except ImportError:
        pass
    
    # Try to import CLASS backend
    try:
        from cloelib.cosmology.class_cosmology import (
            CLASSBackground, CLASSLinearPerturbations, CLASSNonLinearPerturbations
        )
        all_implementations['CLASS'] = {
            'Background': CLASSBackground,
            'LinearPerturbations': CLASSLinearPerturbations,
            'NonLinearPerturbations': CLASSNonLinearPerturbations
        }
    except ImportError:
        pass
    
    # Try to import JAX backend
    try:
        from cloelib.cosmology.jax_cosmology import (
            JAXBackground, JAXLinearPerturbations, JAXNonLinearPerturbations
        )
        all_implementations['JAX'] = {
            'Background': JAXBackground,
            'LinearPerturbations': JAXLinearPerturbations,
            'NonLinearPerturbations': JAXNonLinearPerturbations
        }
    except ImportError:
        pass
    
    # Filter implementations based on selected backend
    if selected_backend is None or selected_backend == "all":
        return all_implementations
    
    # Map CLI option to backend name
    backend_name_map = {
        'camb': 'CAMB',
        'class': 'CLASS', 
        'jax': 'JAX'
    }
    
    backend_name = backend_name_map.get(selected_backend)
    if backend_name and backend_name in all_implementations:
        return {backend_name: all_implementations[backend_name]}
    else:
        return {}


def _get_filtered_backends(config):
    """Get filtered backends based on command line option."""
    selected_backend = config.getoption("--backend") if hasattr(config, 'getoption') else "all"
    return get_backend_implementations(selected_backend)


@pytest.fixture(scope="module")  
def backend_classes(request):
    """
    Parametrized fixture providing cosmology backend classes.
    
    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest fixture request object.
        
    Returns
    -------
    tuple
        (backend_name, backend_classes_dict) where backend_classes_dict contains
        the Background, LinearPerturbations, and NonLinearPerturbations classes.
    """
    # Get filtered implementations based on command line option
    implementations = _get_filtered_backends(request.config)
    
    if not implementations:
        selected_backend = request.config.getoption("--backend")
        pytest.skip(f"Backend '{selected_backend}' is not available or not installed")
    
    # If we have a parametrized backend name, use it; otherwise use the first available
    if hasattr(request, 'param'):
        backend_name = request.param
    else:
        backend_name = list(implementations.keys())[0]
    
    # Note: markers are now added during parametrization in pytest_generate_tests
    
    return backend_name, implementations[backend_name]


# Parametrize the fixture with available backends
def pytest_generate_tests(metafunc):
    """Generate test parameters based on available backends."""
    if "backend_classes" in metafunc.fixturenames:
        # Get the command line backend option
        selected_backend = metafunc.config.getoption("--backend") if hasattr(metafunc.config, 'getoption') else "all"
        implementations = get_backend_implementations(selected_backend)
        
        if implementations:
            # Map backend names to marker names
            backend_marker_map = {
                'CAMB': 'camb',
                'CLASS': 'class',
                'JAX': 'jax'
            }
            
            # Create parametrize values with markers
            param_values = []
            for backend_name in implementations.keys():
                marker_name = backend_marker_map.get(backend_name)
                if marker_name:
                    # Create pytest.param with the appropriate marker
                    param_values.append(
                        pytest.param(backend_name, marks=getattr(pytest.mark, marker_name))
                    )
                else:
                    param_values.append(backend_name)
            
            metafunc.parametrize("backend_classes", param_values, indirect=True)
        else:
            # Skip all tests if no backends available
            metafunc.parametrize("backend_classes", [], indirect=True)


@pytest.fixture(scope="session")
def standard_cosmology():
    """
    Standard cosmology parameters for testing.
    
    Returns
    -------
    dict
        Dictionary of standard cosmological parameters.
    """
    H0 = 67.7
    h = H0/100.
    omch2 = 0.12
    Omega_cdm0 = omch2/h**2
    ombh2 = 0.022
    Omega_b0 = ombh2/h**2
    
    return {
        'H0': H0,
        'Omega_b0': Omega_b0,
        'Omega_cdm0': Omega_cdm0,
        'Omega_k0': 0.,
        'As': 2e-9,
        'ns': 0.96,
        'mnu': 0.,
        'w0': -1.0,
        'wa': 0.0,
        'gamma_MG': 0.0
    }


@pytest.fixture(scope="session") 
def test_redshifts():
    """
    Standard redshift array for testing.
    
    Returns
    -------
    np.ndarray
        Array of test redshifts.
    """
    return np.linspace(0, 2, 20)


@pytest.fixture(scope="session")
def test_wavenumbers():
    """
    Standard wavenumber array for testing.
    
    Returns
    -------
    np.ndarray
        Array of test wavenumbers in units of 1/Mpc.
    """
    return np.logspace(np.log10(1e-4), np.log10(5), 10)


@pytest.fixture(scope="module")
def background_instance(backend_classes, standard_cosmology):
    """
    Create a background cosmology instance for testing.
    
    Parameters
    ----------
    backend_classes : tuple
        (backend_name, backend_classes_dict) from backend_classes fixture.
    standard_cosmology : dict
        Standard cosmological parameters.
        
    Returns
    -------
    Background
        Background cosmology instance.
    """
    backend_name, classes = backend_classes
    BackgroundClass = classes['Background']
    return BackgroundClass(**standard_cosmology)


@pytest.fixture(scope="module")
def perturbation_instances(background_instance, backend_classes, test_redshifts):
    """
    Create perturbation cosmology instances for testing.
    
    Parameters
    ----------
    background_instance : Background
        Background cosmology instance.
    backend_classes : tuple
        (backend_name, backend_classes_dict) from backend_classes fixture.
    test_redshifts : np.ndarray
        Array of test redshifts.
        
    Returns
    -------
    dict
        Dictionary containing 'Linear' and 'NonLinear' perturbation instances.
    """
    backend_name, classes = backend_classes
    LinearClass = classes['LinearPerturbations'] 
    NonLinearClass = classes['NonLinearPerturbations']
    
    # Handle JAX backend differences (no redshifts in constructor)
    if backend_name == 'JAX':
        linear = LinearClass(background=background_instance)
        nonlinear = NonLinearClass(background=background_instance)
    else:
        # CAMB and CLASS backends
        nonlinear_model = 'mead2016' if backend_name == 'CAMB' else 'halofit'
        linear = LinearClass(background=background_instance, redshifts=test_redshifts)
        nonlinear = NonLinearClass(
            background=background_instance, 
            redshifts=test_redshifts,
            nonlinear_model=nonlinear_model
        )
    
    return {"Linear": linear, "NonLinear": nonlinear}
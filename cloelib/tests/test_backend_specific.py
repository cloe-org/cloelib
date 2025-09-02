"""
Backend-specific cosmology tests.

This module contains tests that are specific to individual cosmology backends
and cannot be generalized across all implementations.
"""

import pytest
import numpy as np

# Import backends conditionally
try:
    from cloelib.cosmology.camb_cosmology import (
        CAMBBackground, CAMBLinearPerturbations, CAMBNonLinearPerturbations
    )
    CAMB_AVAILABLE = True
except ImportError:
    CAMB_AVAILABLE = False

try:
    from cloelib.cosmology.class_cosmology import (
        CLASSBackground, CLASSLinearPerturbations, CLASSNonLinearPerturbations
    )
    CLASS_AVAILABLE = True
except ImportError:
    CLASS_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    from cloelib.cosmology.jax_cosmology import (
        JAXBackground, JAXLinearPerturbations, JAXNonLinearPerturbations
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not CAMB_AVAILABLE, reason="CAMB not available")
class TestCAMBSpecific:
    """CAMB-specific tests."""
    
    @pytest.fixture(scope="class")
    def camb_background_instance(self):
        """Create a CAMB background instance."""
        H0 = 67.7
        h = H0/100.
        omch2 = 0.12
        Omega_cdm0 = omch2/h**2
        ombh2 = 0.022
        Omega_b0 = ombh2/h**2
        return CAMBBackground(
            H0=H0, Omega_b0=Omega_b0, Omega_cdm0=Omega_cdm0,
            Omega_k0=0., As=2e-9, ns=0.96, mnu=0.,
            w0=-1.0, wa=0.0, gamma_MG=0.0
        )
    
    @pytest.fixture(scope="class") 
    def camb_zs(self):
        """Redshift array for CAMB tests."""
        return np.linspace(0, 2, 20)
    
    @pytest.fixture(scope="class")
    def camb_perturbation_instances(self, camb_background_instance, camb_zs):
        """Create CAMB perturbation instances with mead2016 nonlinear model."""
        camb_lin = CAMBLinearPerturbations(
            background=camb_background_instance, 
            redshifts=camb_zs
        )
        camb_non = CAMBNonLinearPerturbations(
            background=camb_background_instance, 
            redshifts=camb_zs,
            nonlinear_model='mead2016'
        )
        return {"Linear": camb_lin, "NonLinear": camb_non}
    
    def test_camb_nonlinear_model_mead2016(self, camb_perturbation_instances):
        """Test that CAMB can use the mead2016 nonlinear model."""
        nonlinear_instance = camb_perturbation_instances["NonLinear"]
        # Test that we can call methods without errors
        zs = np.linspace(0, 1, 5)
        ks = np.logspace(-3, 1, 5)
        result = nonlinear_instance.matter_power_spectrum(zs, ks)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(zs), len(ks))


@pytest.mark.skipif(not CLASS_AVAILABLE, reason="CLASS not available")
class TestCLASSSpecific:
    """CLASS-specific tests."""
    
    @pytest.fixture(scope="class")
    def class_background_instance(self):
        """Create a CLASS background instance."""
        H0 = 67.7
        h = H0/100.
        omch2 = 0.12
        Omega_cdm0 = omch2/h**2
        ombh2 = 0.022
        Omega_b0 = ombh2/h**2
        return CLASSBackground(
            H0=H0, Omega_b0=Omega_b0, Omega_cdm0=Omega_cdm0,
            Omega_k0=0., As=2e-9, ns=0.96, mnu=0.,
            w0=-1.0, wa=0.0, gamma_MG=0.0
        )
    
    @pytest.fixture(scope="class") 
    def class_zs(self):
        """Redshift array for CLASS tests."""
        return np.linspace(0, 2, 20)
    
    @pytest.fixture(scope="class")
    def class_perturbation_instances(self, class_background_instance, class_zs):
        """Create CLASS perturbation instances with halofit nonlinear model."""
        class_lin = CLASSLinearPerturbations(
            background=class_background_instance, 
            redshifts=class_zs
        )
        class_non = CLASSNonLinearPerturbations(
            background=class_background_instance, 
            redshifts=class_zs,
            nonlinear_model='halofit'
        )
        return {"Linear": class_lin, "NonLinear": class_non}
    
    def test_class_nonlinear_model_halofit(self, class_perturbation_instances):
        """Test that CLASS can use the halofit nonlinear model."""
        nonlinear_instance = class_perturbation_instances["NonLinear"]
        # Test that we can call methods without errors
        zs = np.linspace(0, 1, 5)
        ks = np.logspace(-3, 1, 5)
        result = nonlinear_instance.matter_power_spectrum(zs, ks)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(zs), len(ks))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXSpecific:
    """JAX-specific tests."""
    
    @pytest.fixture(scope="class")
    def jax_background_instance(self):
        """Create a JAX background instance."""
        H0 = 67.7
        h = H0/100.
        omch2 = 0.12
        Omega_cdm0 = omch2/h**2
        ombh2 = 0.022
        Omega_b0 = ombh2/h**2
        return JAXBackground(
            H0=H0, Omega_b0=Omega_b0, Omega_cdm0=Omega_cdm0,
            Omega_k0=0., As=2e-9, ns=0.96, mnu=0.,
            w0=-1.0, wa=0.0, gamma_MG=0.0
        )
    
    @pytest.fixture(scope="class")
    def jax_zs(self):
        """Redshift array for JAX tests (using JAX numpy)."""
        return jnp.linspace(0, 2, 20)
    
    @pytest.fixture(scope="class")
    def jax_ks(self):
        """Wavenumber array for JAX tests (using JAX numpy)."""
        return jnp.logspace(jnp.log10(1e-4), jnp.log10(5), 10)
    
    @pytest.fixture(scope="class")
    def jax_perturbation_instances(self, jax_background_instance):
        """Create JAX perturbation instances (no redshifts in constructor)."""
        jax_lin = JAXLinearPerturbations(background=jax_background_instance)
        jax_non = JAXNonLinearPerturbations(background=jax_background_instance)
        return {"Linear": jax_lin, "NonLinear": jax_non}
    
    def test_jax_autodiff_hubble_parameter(self, jax_background_instance, jax_zs):
        """Test that JAX background methods are differentiable."""
        
        def hubble_wrapper(z_array):
            return jnp.sum(jax_background_instance.hubble_parameter(z_array, "1/Mpc"))
        
        # Test that we can compute gradients
        grad_fn = jax.grad(hubble_wrapper)
        gradient = grad_fn(jax_zs)
        
        assert isinstance(gradient, jnp.ndarray)
        assert gradient.shape == jax_zs.shape
    
    def test_jax_jit_compilation(self, jax_background_instance, jax_zs):
        """Test that JAX background methods can be JIT compiled."""
        
        @jax.jit
        def jit_omega_m(z_array):
            return jax_background_instance.Omega_m(z_array)
        
        result = jit_omega_m(jax_zs)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == jax_zs.shape
    
    def test_jax_perturbations_no_redshift_constructor(self, jax_perturbation_instances):
        """Test that JAX perturbations don't require redshifts in constructor."""
        # This is tested implicitly by the fixture setup, but let's be explicit
        linear_instance = jax_perturbation_instances["Linear"]
        nonlinear_instance = jax_perturbation_instances["NonLinear"]
        
        # Test that instances were created successfully
        assert linear_instance is not None
        assert nonlinear_instance is not None
        
        # Test that they can still compute power spectra when redshifts are provided to methods
        zs = jnp.array([0.0, 0.5, 1.0])
        ks = jnp.logspace(-2, 1, 5)
        
        linear_pk = linear_instance.matter_power_spectrum(zs, ks)
        nonlinear_pk = nonlinear_instance.matter_power_spectrum(zs, ks)
        
        assert isinstance(linear_pk, jnp.ndarray)
        assert isinstance(nonlinear_pk, jnp.ndarray)
        assert linear_pk.shape == (len(zs), len(ks))
        assert nonlinear_pk.shape == (len(zs), len(ks))
    
    def test_jax_growth_rate_different_signature(self, jax_perturbation_instances, jax_zs):
        """Test that JAX growth_rate takes redshifts as argument."""
        for perturbation_type in ["Linear", "NonLinear"]:
            instance = jax_perturbation_instances[perturbation_type]
            
            # JAX growth_rate takes redshifts as argument
            result = instance.growth_rate(jax_zs)
            assert isinstance(result, jnp.ndarray)
            assert result.ndim == 1
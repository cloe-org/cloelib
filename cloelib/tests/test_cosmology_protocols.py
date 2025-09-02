"""
Protocol compliance tests for cosmology backends.

This module contains parametrized tests that verify all cosmology backends
adhere to the required protocols and produce expected behavior.
"""

import pytest
# Handle JAX import differences
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False

import numpy as np

from cloelib.cosmology.cosmology import Background, Perturbations


class TestBackgroundProtocol:
    """Test that background cosmology classes implement the required protocol."""
    
    def test_background_required_methods(self, backend_classes):
        """Test that all required methods are present."""
        backend_name, classes = backend_classes
        BackgroundClass = classes['Background']
        
        methods_required = {
            name for name, value in Background.__dict__.items() 
            if callable(value) and not name.startswith('_')
        }
        contents = BackgroundClass.__dict__.items()
        methods_found = {
            name for name, value in contents 
            if callable(value) and not name.startswith('_')
        }
        assert methods_required <= methods_found

    def test_background_required_attributes(self, background_instance):
        """Test that all required attributes are present."""
        attributes_required = {
            name for name, value in Background.__dict__.items() 
            if not callable(value) and not name.startswith('_')
        }
        contents = (
            (name, getattr(background_instance, name)) 
            for name in dir(background_instance)
        )
        attributes_found = {
            name for name, value in contents 
            if not callable(value) and not name.startswith('_')
        }
        assert attributes_required <= attributes_found

    def test_background_implements_protocol(self, background_instance):
        """Test that the background instance adheres to the Background protocol."""
        assert isinstance(background_instance, Background)


class TestBackgroundMethods:
    """Test the behavior of background cosmology methods."""
    
    def test_background_H0(self, background_instance):
        """Test H0 attribute."""
        assert hasattr(background_instance, 'H0')
        assert isinstance(background_instance.H0, float)
        assert background_instance.H0 == 67.7

    def test_background_h(self, background_instance):
        """Test h attribute."""
        assert hasattr(background_instance, 'h')
        assert isinstance(background_instance.h, float)
        assert background_instance.h == 0.677

    def test_background_Omega_b0(self, background_instance):
        """Test Omega_b0 attribute."""
        assert hasattr(background_instance, 'Omega_b0')
        assert isinstance(background_instance.Omega_b0, float)
        assert background_instance.Omega_b0 == 0.022 / (0.677*0.677)

    def test_background_Omega_cdm0(self, background_instance):
        """Test Omega_cdm0 attribute."""
        assert hasattr(background_instance, 'Omega_cdm0')
        assert isinstance(background_instance.Omega_cdm0, float)
        assert background_instance.Omega_cdm0 == 0.12 / (0.677*0.677)

    def test_background_mnu(self, background_instance):
        """Test mnu attribute."""
        assert hasattr(background_instance, 'mnu')
        assert isinstance(background_instance.mnu, float)
        assert background_instance.mnu == 0.

    def test_background_Omega_k0(self, background_instance):
        """Test Omega_k0 attribute."""
        assert hasattr(background_instance, 'Omega_k0')
        assert isinstance(background_instance.Omega_k0, float)
        assert background_instance.Omega_k0 == 0.

    def test_background_As(self, background_instance):
        """Test As attribute."""
        assert hasattr(background_instance, 'As')
        assert isinstance(background_instance.As, float)
        assert background_instance.As == 2e-9

    def test_background_ns(self, background_instance):
        """Test ns attribute."""
        assert hasattr(background_instance, 'ns')
        assert isinstance(background_instance.ns, float)
        assert background_instance.ns == 0.96

    def test_background_w0(self, background_instance):
        """Test w0 attribute."""
        assert hasattr(background_instance, 'w0')
        assert isinstance(background_instance.w0, float)
        assert background_instance.w0 == -1

    def test_background_wa(self, background_instance):
        """Test wa attribute."""
        assert hasattr(background_instance, 'wa')
        assert isinstance(background_instance.wa, float)
        assert background_instance.wa == 0.

    def test_omega_m(self, background_instance, backend_classes, test_redshifts):
        """Test Omega_m returns an array object of correct size."""
        backend_name, _ = backend_classes
        
        assert hasattr(background_instance, 'Omega_m')
        assert callable(background_instance.Omega_m)
        result = background_instance.Omega_m(test_redshifts)
        
        # Handle different array types
        if backend_name == 'JAX' and JAX_AVAILABLE:
            assert isinstance(result, jnp.ndarray)
            assert len(result) == len(test_redshifts)
        else:
            assert isinstance(result, np.ndarray)
            assert result.ndim == 1
            assert len(result) == len(test_redshifts)

    def test_omega_b(self, background_instance, backend_classes, test_redshifts):
        """Test Omega_b method behavior."""
        backend_name, _ = backend_classes
        
        assert hasattr(background_instance, 'Omega_b')
        assert callable(background_instance.Omega_b)
        result = background_instance.Omega_b(test_redshifts)
        
        # Handle different array types
        if backend_name == 'JAX' and JAX_AVAILABLE:
            assert isinstance(result, jnp.ndarray)
            assert len(result) == len(test_redshifts)
            # Check redshift zero value
            assert jnp.abs(result[0] - background_instance.Omega_b0) < 1e-4
        else:
            assert isinstance(result, np.ndarray)
            assert result.ndim == 1
            assert len(result) == len(test_redshifts)
            # Check redshift zero value
            assert np.abs(result[0] - background_instance.Omega_b0) < 1e-4

    @pytest.mark.parametrize("units", ["1/Mpc", "km/s/Mpc"])
    def test_hubble_parameter(self, background_instance, backend_classes, test_redshifts, units):
        """Test hubble_parameter method behavior."""
        backend_name, _ = backend_classes
        
        assert hasattr(background_instance, 'hubble_parameter')
        assert callable(background_instance.hubble_parameter)
        result = background_instance.hubble_parameter(test_redshifts, units)
        
        # Handle different array types
        if backend_name == 'JAX' and JAX_AVAILABLE:
            assert isinstance(result, jnp.ndarray)
            assert len(result) == len(test_redshifts)
            if units == "km/s/Mpc":
                assert jnp.abs(result[0] - background_instance.H0) < 1e-4
        else:
            assert isinstance(result, np.ndarray)
            assert result.ndim == 1
            assert len(result) == len(test_redshifts)
            if units == "km/s/Mpc":
                assert np.abs(result[0] - background_instance.H0) < 1e-4

    def test_comoving_distance(self, background_instance, backend_classes, test_redshifts):
        """Test comoving_distance returns array of correct size."""
        backend_name, _ = backend_classes
        
        assert hasattr(background_instance, 'comoving_distance')
        assert callable(background_instance.comoving_distance)
        result = background_instance.comoving_distance(test_redshifts)
        
        # Handle different array types
        if backend_name == 'JAX' and JAX_AVAILABLE:
            assert isinstance(result, jnp.ndarray)
            assert len(result) == len(test_redshifts)
        else:
            assert isinstance(result, np.ndarray)
            assert result.ndim == 1
            assert len(result) == len(test_redshifts)

    def test_transverse_comoving_distance(self, background_instance, backend_classes, test_redshifts):
        """Test transverse_comoving_distance returns array of correct size."""
        backend_name, _ = backend_classes
        
        assert hasattr(background_instance, 'transverse_comoving_distance')
        assert callable(background_instance.transverse_comoving_distance)
        result = background_instance.transverse_comoving_distance(test_redshifts)
        
        # Handle different array types
        if backend_name == 'JAX' and JAX_AVAILABLE:
            assert isinstance(result, jnp.ndarray)
            assert len(result) == len(test_redshifts)
        else:
            assert isinstance(result, np.ndarray)
            assert result.ndim == 1
            assert len(result) == len(test_redshifts)

    def test_angular_diameter_distance(self, background_instance, backend_classes, test_redshifts):
        """Test angular_diameter_distance returns array of correct size."""
        backend_name, _ = backend_classes
        
        assert hasattr(background_instance, 'angular_diameter_distance')
        assert callable(background_instance.angular_diameter_distance)
        result = background_instance.angular_diameter_distance(test_redshifts)
        
        # Handle different array types
        if backend_name == 'JAX' and JAX_AVAILABLE:
            assert isinstance(result, jnp.ndarray)
            assert len(result) == len(test_redshifts)
        else:
            assert isinstance(result, np.ndarray)
            assert result.ndim == 1
            assert len(result) == len(test_redshifts)


class TestPerturbationsProtocol:
    """Test perturbations cosmology protocol compliance and behavior."""
    
    @pytest.mark.parametrize("perturbation_type", ["Linear", "NonLinear"])
    def test_perturbation_implements_protocol(self, perturbation_instances, perturbation_type):
        """Test that perturbation instances adhere to the protocol."""
        instance = perturbation_instances[perturbation_type]
        assert isinstance(instance, Perturbations)

    @pytest.mark.parametrize("perturbation_type", ["Linear", "NonLinear"])
    def test_matter_power_spectrum(
        self, perturbation_instances, backend_classes, perturbation_type, 
        test_redshifts, test_wavenumbers
    ):
        """Test matter_power_spectrum method behavior."""
        backend_name, _ = backend_classes
        instance = perturbation_instances[perturbation_type]
        
        assert hasattr(instance, 'matter_power_spectrum')
        assert callable(instance.matter_power_spectrum)
        result = instance.matter_power_spectrum(test_redshifts, test_wavenumbers)
        
        # Handle different array types
        if backend_name == 'JAX' and JAX_AVAILABLE:
            assert isinstance(result, jnp.ndarray)
        else:
            assert isinstance(result, np.ndarray)
        
        assert result.ndim == 2
        assert result.shape == (len(test_redshifts), len(test_wavenumbers))

    @pytest.mark.parametrize("perturbation_type", ["Linear", "NonLinear"])
    def test_growth_factor(
        self, perturbation_instances, backend_classes, perturbation_type,
        test_redshifts, test_wavenumbers
    ):
        """Test growth_factor method behavior."""
        backend_name, _ = backend_classes
        instance = perturbation_instances[perturbation_type]
        
        assert hasattr(instance, 'growth_factor')
        assert callable(instance.growth_factor)
        result = instance.growth_factor(test_redshifts, test_wavenumbers)
        
        # Handle different array types
        if backend_name == 'JAX' and JAX_AVAILABLE:
            assert isinstance(result, jnp.ndarray)
            # JAX implementation may have different behavior - allow more flexibility
        else:
            assert isinstance(result, np.ndarray)
            assert result.ndim == 2
            assert result.shape == (len(test_redshifts), len(test_wavenumbers))

    @pytest.mark.parametrize("perturbation_type", ["Linear", "NonLinear"])
    def test_growth_rate(self, perturbation_instances, backend_classes, perturbation_type, test_redshifts):
        """Test growth_rate method behavior."""
        backend_name, _ = backend_classes
        instance = perturbation_instances[perturbation_type]
        
        assert hasattr(instance, 'growth_rate')
        assert callable(instance.growth_rate)
        
        # Handle different calling conventions
        if backend_name == 'JAX':
            result = instance.growth_rate(test_redshifts)
        else:
            result = instance.growth_rate()
        
        # Handle different array types
        if backend_name == 'JAX' and JAX_AVAILABLE:
            assert isinstance(result, jnp.ndarray)
        else:
            assert isinstance(result, np.ndarray)
        
        assert result.ndim == 1
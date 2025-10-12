"""Tests for MGrowthLinearPerturbations class."""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.cosmology.mgrowth_cosmology import MGrowthLinearPerturbations


@pytest.fixture
def mock_background():
    """Create a mock background instance."""
    background = Mock(spec=Background)
    background.Omega_k0 = 0.0
    background.h = 0.677
    background.H0 = 67.7
    background.Omega_cdm0 = 0.25
    background.Omega_b0 = 0.05
    background.mnu = 0.0
    background.w0 = -1.0
    background.wa = 0.0
    
    # Mock methods
    background.Omega_m.return_value = np.array([0.3])
    return background


@pytest.fixture
def mock_base_perturbations():
    """Create a mock base perturbations instance."""
    perturbations = Mock(spec=Perturbations)
    perturbations.z = np.array([0.0, 0.5, 1.0, 2.0])
    perturbations.k = np.logspace(-3, 1, 50)  # 50 k values
    
    # Mock matter_power_spectrum method
    def mock_power_spectrum(z, k):
        return np.ones_like(k) * 1e4  # Simple constant power spectrum
    
    perturbations.matter_power_spectrum = mock_power_spectrum
    return perturbations


@pytest.fixture
def mock_mgrowth():
    """Mock MGrowth module."""
    with patch('cloelib.cosmology.mgrowth_cosmology.mgrowth') as mock_mg:
        # Mock MGrowth classes
        mock_w0wacdm = Mock()
        mock_fr = Mock()
        mock_dgp = Mock()
        mock_ide = Mock()
        mock_gamma = Mock()
        mock_gammaz = Mock()
        mock_musigma = Mock()
        
        # Mock growth_parameters method for all models
        def mock_growth_params(**kwargs):
            # Return mock growth data with correct shape
            # For most models: (z_len,) arrays that get repeated for k
            z_len = 256
            D = np.ones(z_len)
            f = np.ones(z_len) * 0.5
            return D, f
        
        # Special case for f(R) which returns k-dependent growth
        def mock_growth_params_fr(**kwargs):
            z_len = 256
            k_len = 128
            # f(R) returns (k_len, z_len) which gets transposed to (z_len, k_len)
            D = np.ones((k_len, z_len))
            f = np.ones((k_len, z_len)) * 0.5
            return D, f
        
        # Assign appropriate mock functions
        mock_w0wacdm.growth_parameters = mock_growth_params
        mock_fr.growth_parameters = mock_growth_params_fr
        mock_dgp.growth_parameters = mock_growth_params
        mock_ide.growth_parameters = mock_growth_params
        mock_gamma.growth_parameters = mock_growth_params
        mock_gammaz.growth_parameters = mock_growth_params
        mock_musigma.growth_parameters = mock_growth_params
        
        mock_mg.w0waCDM = Mock(return_value=mock_w0wacdm)
        mock_mg.fR_HS = Mock(return_value=mock_fr)
        mock_mg.nDGP = Mock(return_value=mock_dgp)
        mock_mg.IDE = Mock(return_value=mock_ide)
        mock_mg.Linder_gamma = Mock(return_value=mock_gamma)
        mock_mg.Linder_gamma_a = Mock(return_value=mock_gammaz)
        mock_mg.mu_a = Mock(return_value=mock_musigma)
        
        yield mock_mg


class TestMGrowthLinearPerturbations:
    """Test class for MGrowthLinearPerturbations."""
    
    def test_init_w0wacdm_model(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test initialization with w0wacdm model."""
        mgpars = {}
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars=mgpars
        )
        
        assert mg_pert.background == mock_background
        assert mg_pert.base == mock_base_perturbations
        assert mg_pert.gravity_model == 'w0wacdm'
        assert mg_pert.mgpars == mgpars
        assert hasattr(mg_pert, 'z_sorted')
        assert hasattr(mg_pert, 'k')
        assert hasattr(mg_pert, 'dz_interp')
        assert hasattr(mg_pert, 'fz_interp')
    
    def test_init_fr_model(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test initialization with f(R) model."""
        mgpars = {'fr0': 1e-5}
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='fr',
            mgpars=mgpars
        )
        
        assert mg_pert.gravity_model == 'fr'
        assert mg_pert.mgpars == mgpars
    
    def test_init_dgp_model(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test initialization with DGP model."""
        mgpars = {'omega_rc': 0.1}
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='dgp',
            mgpars=mgpars
        )
        
        assert mg_pert.gravity_model == 'dgp'
        assert mg_pert.mgpars == mgpars
    
    def test_init_ide_model(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test initialization with IDE model."""
        mgpars = {'xi': 0.1}
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='ide',
            mgpars=mgpars
        )
        
        assert mg_pert.gravity_model == 'ide'
        assert mg_pert.mgpars == mgpars
    
    def test_init_gamma_model(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test initialization with gamma model."""
        mgpars = {'gamma0': 0.55}
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='gamma',
            mgpars=mgpars
        )
        
        assert mg_pert.gravity_model == 'gamma'
        assert mg_pert.mgpars == mgpars
    
    def test_init_gammaz_model(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test initialization with gammaz model."""
        mgpars = {'gamma0': 0.55, 'gamma1': 0.1}
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='gammaz',
            mgpars=mgpars
        )
        
        assert mg_pert.gravity_model == 'gammaz'
        assert mg_pert.mgpars == mgpars
    
    def test_init_musigma_de_model(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test initialization with mu-sigma DE model."""
        mgpars = {'mu0': 0.1, 'sigma0': 0.05}
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='musigma-de',
            mgpars=mgpars
        )
        
        assert mg_pert.gravity_model == 'musigma-de'
        assert mg_pert.mgpars == mgpars
        assert hasattr(mg_pert, 'mu_interp')
        assert hasattr(mg_pert, 'sigma_lensing')
    
    def test_init_non_flat_geometry_error(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test that non-flat geometry raises an error."""
        mock_background.Omega_k0 = 0.1  # Non-flat
        
        with pytest.raises(AssertionError, match="Non flat geometries not supported"):
            MGrowthLinearPerturbations(
                background=mock_background,
                base_linear_perturbations=mock_base_perturbations,
                gravity_model='w0wacdm',
                mgpars={}
            )
    
    def test_init_unsupported_gravity_model(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test that unsupported gravity model raises an error."""
        with pytest.raises(ValueError, match="Unsupported gravity model"):
            MGrowthLinearPerturbations(
                background=mock_background,
                base_linear_perturbations=mock_base_perturbations,
                gravity_model='unsupported_model',
                mgpars={}
            )
    
    def test_init_musigma_missing_parameters(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test that missing mu-sigma parameters raise an error."""
        with pytest.raises(ValueError, match="Mu-Sigma parameters are not properly specified"):
            MGrowthLinearPerturbations(
                background=mock_background,
                base_linear_perturbations=mock_base_perturbations,
                gravity_model='musigma-de',
                mgpars={'mu0': 0.1}  # Missing sigma0
            )
    
    def test_growth_factor_single_values(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test growth_factor method with single values."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        z = 1.0
        k = 0.1
        result = mg_pert.growth_factor(z, k)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)
        assert result[0, 0] > 0  # Growth factor should be positive
    
    def test_growth_factor_array_inputs(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test growth_factor method with array inputs."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        zs = np.array([0.0, 0.5, 1.0])
        ks = np.array([0.01, 0.1, 1.0])
        result = mg_pert.growth_factor(zs, ks)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        assert np.all(result > 0)  # All growth factors should be positive
    
    def test_growth_rate_single_values(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test growth_rate method with single values."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        z = 1.0
        k = 0.1
        result = mg_pert.growth_rate(z, k)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)
        assert result[0, 0] > 0  # Growth rate should be positive
    
    def test_growth_rate_array_inputs(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test growth_rate method with array inputs."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        zs = np.array([0.0, 0.5, 1.0])
        ks = np.array([0.01, 0.1, 1.0])
        result = mg_pert.growth_rate(zs, ks)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        assert np.all(result > 0)  # All growth rates should be positive
    
    def test_matter_power_spectrum_single_values(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test matter_power_spectrum method with single values."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        z = 1.0
        k = 0.1
        result = mg_pert.matter_power_spectrum(z, k)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)
        assert result[0, 0] > 0  # Power spectrum should be positive
    
    def test_matter_power_spectrum_array_inputs(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test matter_power_spectrum method with array inputs."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        zs = np.array([0.0, 0.5, 1.0])
        ks = np.array([0.01, 0.1, 1.0])
        result = mg_pert.matter_power_spectrum(zs, ks)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 3)
        assert np.all(result > 0)  # All power spectrum values should be positive
    
    def test_compute_mu_de_interp(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test the _compute_mu_de_interp method."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='musigma-de',
            mgpars={'mu0': 0.1, 'sigma0': 0.05}
        )
        
        # Test the interpolation function
        mu_interp = mg_pert._compute_mu_de_interp(0.1, 0.3, -1.0, 0.0)
        
        assert callable(mu_interp)
        
        # Test interpolation at some scale factors
        a_test = np.array([0.1, 0.5, 1.0])
        mu_values = mu_interp(a_test)
        
        assert isinstance(mu_values, np.ndarray)
        assert len(mu_values) == 3
        assert np.all(mu_values > 0)  # mu should be positive
    
    def test_compute_sigma_de_interp(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test the _compute_sigma_de_interp method."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='musigma-de',
            mgpars={'mu0': 0.1, 'sigma0': 0.05}
        )
        
        # Test the interpolation function
        sigma_interp = mg_pert._compute_sigma_de_interp(0.05, 0.3, -1.0, 0.0)
        
        assert callable(sigma_interp)
        
        # Test interpolation at some redshifts
        z_test = np.array([0.0, 1.0, 2.0])
        sigma_values = sigma_interp(z_test)
        
        assert isinstance(sigma_values, np.ndarray)
        assert len(sigma_values) == 3
        assert np.all(sigma_values > 0)  # sigma should be positive
    
    def test_growth_factor_normalization(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test that growth factor is normalized to D(0) = 1."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        # At z=0, growth factor should be 1 (normalized)
        z0 = 0.0
        k = 0.1
        result = mg_pert.growth_factor(z0, k)
        
        assert abs(result[0, 0] - 1.0) < 1e-10
    
    def test_redshift_sorting(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test that redshifts are properly sorted."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        # Check that z_sorted is in ascending order
        assert np.all(np.diff(mg_pert.z_sorted) > 0)
        
        # Check that a_sorted is in ascending order (early to late times)
        assert np.all(np.diff(mg_pert.a_sorted) > 0)
    
    def test_k_array_properties(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test properties of the k array."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        # Check that k array is log-spaced
        assert len(mg_pert.k) == 128
        assert mg_pert.k[0] == mock_base_perturbations.k[0]
        assert mg_pert.k[-1] == mock_base_perturbations.k[-1]
        assert np.all(np.diff(np.log(mg_pert.k)) > 0)  # Log-spaced
    
    def test_musigma_check_ranges(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test the check_ranges flag for mu-sigma model."""
        # Test case where mu0 > 2*sigma0 + 1
        mgpars = {'mu0': 1.2, 'sigma0': 0.05}  # 1.2 > 2*0.05 + 1 = 1.1, so False
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='musigma-de',
            mgpars=mgpars
        )
        
        assert mg_pert.check_ranges == False
        
        # Test case where mu0 <= 2*sigma0 + 1
        mgpars = {'mu0': 0.1, 'sigma0': 0.05}  # 0.1 <= 2*0.05 + 1 = 1.1, so True
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='musigma-de',
            mgpars=mgpars
        )
        
        assert mg_pert.check_ranges == True
    
    def test_background_dict_construction(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test that background dictionary is constructed correctly."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        # The background dict should be constructed in __init__
        # We can't directly access it, but we can verify the MGrowth cosmology was created
        assert hasattr(mg_pert, 'mg_cosmo')
        assert mg_pert.mg_cosmo is not None
    
    def test_interpolator_attributes(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test that all required interpolators are created."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        # Check that all interpolators exist
        assert hasattr(mg_pert, 'dz_interp')
        assert hasattr(mg_pert, 'dz_norm_dz0_interp')
        assert hasattr(mg_pert, 'fz_interp')
        assert hasattr(mg_pert, 'dz_norm_lcdm_interp')
        
        # Check that they are callable
        assert callable(mg_pert.dz_interp)
        assert callable(mg_pert.dz_norm_dz0_interp)
        assert callable(mg_pert.fz_interp)
        assert callable(mg_pert.dz_norm_lcdm_interp)


class TestMGrowthLinearPerturbationsEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_negative_redshifts(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test behavior with negative redshifts."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        # Test with negative redshift (should extrapolate)
        z = -0.1
        k = 0.1
        result = mg_pert.growth_factor(z, k)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 1)
    
    def test_extreme_k_values(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test behavior with extreme k values."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        # Test with very small k
        z = 1.0
        k_small = 1e-6
        result_small = mg_pert.growth_factor(z, k_small)
        
        # Test with very large k
        k_large = 1e3
        result_large = mg_pert.growth_factor(z, k_large)
        
        assert isinstance(result_small, np.ndarray)
        assert isinstance(result_large, np.ndarray)
        assert result_small.shape == (1, 1)
        assert result_large.shape == (1, 1)
    
    def test_empty_arrays(self, mock_background, mock_base_perturbations, mock_mgrowth):
        """Test behavior with empty arrays."""
        mg_pert = MGrowthLinearPerturbations(
            background=mock_background,
            base_linear_perturbations=mock_base_perturbations,
            gravity_model='w0wacdm',
            mgpars={}
        )
        
        # Test with empty arrays
        zs = np.array([])
        ks = np.array([])
        result = mg_pert.growth_factor(zs, ks)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 0)


if __name__ == "__main__":
    pytest.main([__file__])

#import jax.numpy as np
import numpy as np

from numpy.testing import assert_raises, assert_equal, assert_allclose

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.clustering import HaloClustering


def _test_clustering(CL,perturbations):
    z_test = np.array([0., 1.])
    r_test = np.array([30.,60.,90.])
    lob_test = np.array([50.]) 

    print("    APcorr_func")
    ref_APcorr = np.array([1.0162  , 1.016033]) 
    assert_allclose(CL.APcorr_func(z_test), ref_APcorr,rtol=1e-04)

    
    print("    WF_ra")
    ref_WF_ra0 = np.array([[[9.9962026e-01, 5.3620315e-04],
                            [9.9894708e-01, 3.8502066e-04]],
                           [[9.9957979e-01, 5.4349063e-04],
                            [9.9900585e-01, 3.8453474e-04]]])
    ref_WF_ra1 = np.array([[ 830784.512318, 2254986.533435],
                           [ 830373.221984, 2253870.173956]])


    WF,VF = CL.WF_ra(z_test, r_test)
    assert_allclose(WF, ref_WF_ra0,rtol=1e-04)    
    assert_allclose(VF, ref_WF_ra1,rtol=1e-04)

    
    print("    Pk_IR_func")
    ref_Pk_IR = np.array([[3782.7314, 63.855637],
                          [1398.2388,  23.673042]])

    Pk_test = perturbations.matter_power_spectrum(z_test, CL.k,
                                                  hubble_units=True, k_hunit=True)
    assert_allclose(CL.Pk_IR_func(Pk_test), ref_Pk_IR, rtol=1e-4)

    
    print("    photoz_rsd_correction")
    ref_phz_rsd_0 = np.array([[0.99999981, 0.84024509],
                              [0.99939964, 0.02087668]])
    ref_phz_rsd_1 = np.array([[6.666667e-01, 2.524346e-01],
                              [5.807121e-01, 1.010152e-05]])
    ref_phz_rsd_2 = np.array([[2.000000e-01, 3.714264e-02],
                              [1.518641e-01, 3.665835e-09]])

    sigma_zob = 0.025 * z_test + 5e-6 * lob_test
    corr0,corr1,corr2 = CL.photoz_rsd_correction(z_test, sigma_zob)
    assert_allclose(corr0, ref_phz_rsd_0, rtol=1e-04)
    assert_allclose(corr1, ref_phz_rsd_1, rtol=1e-04)
    assert_allclose(corr2, ref_phz_rsd_2, rtol=1e-04)



def test_clustering():
    # Cosmology parameters
    print("# Cosmology parameters")
    _H0 = 67.7
    _h = _H0 / 100.0
    _omch2 = 0.12
    _ombh2 = 0.022
    _ns = 0.96
    _mnu = 0.06
    _As = 2.0e-9
    _cosmo_pars = dict(
        H0=_H0,
        Omega_cdm0=_omch2 / _h**2,
        Omega_b0=_ombh2 / _h**2,
        Omega_k0=0.0,
        w0=-1.0,
        wa=0.0,
        ns=_ns,
        mnu=_mnu,
        As=_As,
        gamma_MG=0.0,
    )

    background = CAMBBackground(**_cosmo_pars)
    perturbations = CAMBLinearPerturbations(background, np.linspace(0.0, 2.0, 100))

    _cosmo_pars_fid = {**_cosmo_pars}
    _cosmo_pars_fid["H0"] = 73.0
    background_fid = CAMBBackground(**_cosmo_pars_fid)
    perturbations_fid = CAMBLinearPerturbations(
        background_fid, np.linspace(0.0, 2.0, 100)
    )
    k_min = 1e-3
    k_max = 1e0
    k_div = 2
    nonu = True
    CL = HaloClustering(perturbations, perturbations_fid, nonu, k_div, k_min, k_max)
    _test_clustering(CL,perturbations)

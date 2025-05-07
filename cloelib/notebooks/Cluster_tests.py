from cloelib.observables.clusters.selection_function import SelectionFunction
from cloelib.observables.clusters.halo_statistics import HaloStatistics, HaloStatisticsTinker, HaloStatisticsCastro
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.clusters.clustering import HaloClustering

import numpy as np

# Cosmology parameters
H0 =  67.7
h = H0/100.
sigma8 = 0.8277
omch2 = 0.12
Omega_cdm0 = omch2/h**2
ombh2 = 0.022
Omega_b0 = ombh2/h**2
Omega_k0 = 0.
w = -1.
wa = 0.
ns = 0.96
mnu = 0.
As=2e-9

# sel. function parameters
A_l = 0.5
B_l = 0.6
C_l = 0.5
sig_A_l = 0.1
sig_B_l = 0.
sig_C_l = 0.
sig_lambda_norm = 0.1
sig_lambda_z = 0.1
sig_lambda_exponent = 0.1
sig_z_z = 0.1
sig_z_lambda = 0.1


background = CAMBBackground(H0=H0, Omega_b0=Omega_b0, Omega_cdm0=Omega_cdm0, 
                            Omega_k0=Omega_k0,
                            As=As, ns=ns, mnu=0., w0=-1.0, wa=0.0, 
                            gamma_MG=0.0)
perturbations = CAMBLinearPerturbations(background, np.linspace(0., 2., 100))


# SelectionFunction
SF = SelectionFunction(A_l, B_l, C_l, sig_A_l, sig_B_l, sig_C_l, sig_lambda_norm, sig_lambda_z, sig_lambda_exponent, sig_z_z, sig_z_lambda)

z_test = np.linspace(0.01, 1., 20)
zob_test = np.linspace(0.1, 1.1, 20)
M_test = 1.e14
l_test = np.logspace(0., 2., 20)
lob_test = np.logspace(0.2, 2.2, 20)

SF.lnlambda(z_test, M_test)
SF.scatter_lnl(z_test, M_test)
SF.P_lnlbd(z_test, M_test, l_test)
SF.scatter_lbdobs_lbd(z_test, l_test)
SF.P_lbdobs_lbd(z_test, l_test, lob_test)
SF.scatter_zobs_z(lob_test, z_test)
SF.P_zobs_z(zob_test, lob_test, z_test)

# HaloStatistics
HS = HaloStatistics(perturbations, 'vir')

k_test = np.logspace(-2, 1, 100)
R_test = np.logspace(-1, 1, 20)
M_test = np.logspace(14, 15, 50)

HS.window(k_test, R_test)
HS.radius_M(M_test)
HS.delta_c(z_test)
HS.get_Delta_crit(z_test)
HS.sigma_z_R(z_test, R_test)
HS.sigma_z_M(z_test, M_test)
HS.nu_z_M(z_test, M_test)
HS.dlns_dlnR(z_test, M_test)

HS_tinker = HaloStatisticsTinker(perturbations, 'vir')
#HS_tinker.bias(z_test, M_test)

HS_castro = HaloStatisticsCastro(perturbations, 'vir')
HS_castro.dn_dm(z_test, M_test)
HS_castro.bias(z_test, M_test)



# clustering

H0_fid = 73.
background_fid = CAMBBackground(H0=H0_fid, Omega_b0=Omega_b0, Omega_cdm0=Omega_cdm0,
                            Omega_k0=Omega_k0,
                            As=As, ns=ns, mnu=0., w0=-1.0, wa=0.0,
                            gamma_MG=0.0)
perturbations_fid = CAMBLinearPerturbations(background_fid, np.linspace(0., 2., 100))


k_min = 1e-4
k_max = 2e0
k_div = 300
nonu  = True
CL = HaloClustering(perturbations,perturbations_fid,nonu,k_div,k_min,k_max)

z_test = np.linspace(0.,2.,20)
r_test = np.geomspace(20,150,30)

CL.APcorr_func(z_test)
CL.WF_ra(z_test,r_test)


Pk_test = perturbations.matter_power_spectrum(
            z_test, CL.k, hubble_units=True, k_hunit=True)
CL.Pk_IR_func(Pk_test)


sigma_zob = SF.scatter_zobs_z(lob_test, z_test)
CL.photoz_rsd_correction(z_test, sigma_zob)






from cloelib.observables.clusters.selection_function import SelectionFunction
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations

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

#
camb_instance = CAMBBackground(H0=H0, Omega_b0=Omega_b0, Omega_cdm0=Omega_cdm0, 
                               Omega_k0=Omega_k0,
                               As=As, ns=ns, mnu=0., w0=-1.0, wa=0.0, 
                               gamma_MG=0.0)

# SelecitonFunction
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



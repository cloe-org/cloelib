# General imports
import numpy as np

# cloelite imports: we load the following interfaces
from cloelib.cosmology.mochi_class_cosmology import (
    mochiCLASSBackground,
)  # requires cloelib branch 292-mochi-class

# for solving M2 ODE (Lombriser parameterisation)
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline


def Solve_M2_ODE(
    lna_array,
    H,
    rho_phi,
    Dkin,
    cs2,
    b,  # stable basis
    M2_init=1.0,
    M2D_init=0.0,  # initial conditions for ODE
    w0=-1,
    wa=0,
):
    """
    solve ODE for M2 (= M^2) numerically
    M2'' + r (M2')^2/M2 + p M2' + q M2 = g
    with r,p,q,g as functions of x=lna
    solve forward, initioal conditions at lna=-5
    """
    # H_dot = dH/dlna
    H_spline = UnivariateSpline(lna_array, H, s=0)
    H_dot = H_spline.derivative()(lna_array)

    # Calculate w(a) using the w0wa model
    w_array = w0 + wa * (1 - np.exp(lna_array))

    # Calculate coefficients
    # derived from mochi cs2 eqn
    r = 1 / b - 2
    p_array = 1 - b + H_dot / H
    q_array = b * H_dot / H + b / 2 * cs2 * Dkin
    g_array = 3 * (
        b / (2 * H**2) * (rho_phi + w_array * rho_phi) + 1 / 3 * b * H_dot / H
    )

    # Lombriser 2018 paper -> tested, equivalent to eqn above
    """
    r = (1 - 2 * b) / b
    p_array = -1/2 * (1 + 2 * b + 3 * Omega_DE * w_array)
    q_array = b/2 * (cs2 * Dkin - 3 - 3 * Omega_DE * w_array)
    g_array = b/2 * (3 * Omega_DE - 3)
    """

    # Interpolation functions for p, q, g
    def p(lna):
        return np.interp(lna, lna_array, p_array)

    def q(lna):
        return np.interp(lna, lna_array, q_array)

    def g(lna):
        return np.interp(lna, lna_array, g_array)

    # Define the ODE system: M2' = D, D' = [Original ODE in terms of M2 and D]
    def system(lna, Y):
        M2, D = Y  # Y is the vector [M2, M2']
        dM2_dlna = D
        dD_dlna = g(lna) - q(lna) * M2 - p(lna) * D - r * (D**2) / M2
        return [dM2_dlna, dD_dlna]

    # Initial conditions
    # Set your initial condition for M2 and M2'
    initial_conditions = [M2_init, M2D_init]
    # initial_conditions = [1, 1]

    # Solve the ODE
    try:
        solution = solve_ivp(
            system,
            t_span=(lna_array[0], lna_array[-1]),
            y0=initial_conditions,
            t_eval=lna_array,
            method="RK45",
            rtol=1e-7,
            atol=1e-10,
        )  # higher accuracy crucial, cause small deviations will be magnified by 1/Dkin later on
        print("size lna_array", len(lna_array))
        print("size M2 array", len(solution.y[0]))

        if solution.status == -1:
            raise ValueError("Integration failed.")

    except ValueError:
        raise

    return solution


def stable_basis_from_M2_ODE(
    H0,
    Omega_b0,
    Omega_cdm0,
    Omega_k0,
    As,
    ns,  # cosmological parameters
    lna_mochi,  # lna range for basis functions
    w0,
    wa,  # background
    b=2,  # M2 parameterisation, b is either one or two
    s=0,  # cs2 parameterisation
    a0=0,
    a1=0,
    u1=1,
    u2=1,  # alpha parameterisation
    M2_init=1,  # initial condition for M2 at lna=-5
    M2D_init=0,  # initial condition for M2' at lna=-5
    # cosmological parameters
):
    """
    Lombriser et al 2018 parameterisation of stable basis functions (eqn 4.4-4.7)
    alpha is equiv to Dkin
    solve M2 ODE to incorporate alpha_B ~ alpha_M proportionality
    here alpha_M = d ln M2/ d ln a = 1/M2 d M2/ d ln a = M2' / M2
    lna_mochi: lna range for basis functions - here using internal mochi spacing

    output:
        dictionary with stable basis functions, to be fed into mochi_class cloelib protocol
    """

    ###############################################
    # get Omega_DE from background - ONLY FOR wowa
    ###############################################
    stable_MG_dict = {
        # "w0": w0,
        # "wa": wa,
    }

    # background for w0wa (MG off)
    background_instance = mochiCLASSBackground(
        H0=H0,
        Omega_b0=Omega_b0,
        Omega_cdm0=Omega_cdm0,
        Omega_k0=Omega_k0,
        As=As,
        ns=ns,
        mnu=0.0,
        N_mnu=0,
        w0=w0,
        wa=wa,
        gamma_MG=0.55,
        mg_stable_basis_on=False,
        mg_background_model="wowa",
        stable_MG_dict=stable_MG_dict,
    )

    # compute Omega_DE from background for parameterisation, and get background evolution for M2 ODE
    background_wowa = background_instance.get_background()
    lna_wowa = np.log(
        1 / (1 + background_wowa["z"])
    )  # lna-spacing for all time functions, output by mochi

    # rho_fld if just wowa (no mg)
    H_wowa_lna = background_wowa["H [1/Mpc]"]  # Hubble in 1/Mpc
    rhoDE_wowa_lna = background_wowa[
        "(.)rho_fld"
    ]  # only valid when flg used in mochi background instance (i.e. no-MG)
    rhoCrit_wowa_lna = background_wowa["(.)rho_crit"]
    Omega_DE_lna = rhoDE_wowa_lna / rhoCrit_wowa_lna

    # interpolate for late times (MG scale factors)
    Omega_DE = UnivariateSpline(lna_wowa, Omega_DE_lna, s=0)(lna_mochi)
    H_wowa = UnivariateSpline(lna_wowa, H_wowa_lna, s=0)(lna_mochi)
    rhoDE_wowa = UnivariateSpline(lna_wowa, rhoDE_wowa_lna, s=0)(lna_mochi)

    ###############################################
    ####### parametertisation stable basis ########
    ###############################################
    cs2_param = 1 + s * Omega_DE / Omega_DE[-1]
    alpha_param = (
        (a0 * (1 + s) + a1 * (1 - np.exp(lna_mochi) ** u1))
        * (Omega_DE / Omega_DE[-1]) ** u2
        / cs2_param
    )

    ############# compute M2 from ODE #############
    if b == 0:
        M2_param = np.ones_like(cs2_param)
        alpha_M = np.zeros_like(cs2_param)
        alpha_B_propto = np.zeros_like(cs2_param)
    else:
        M2_solution = Solve_M2_ODE(
            lna_mochi,
            H_wowa,
            rhoDE_wowa,
            alpha_param,
            cs2_param,
            b,
            w0=w0,
            wa=wa,
            M2_init=M2_init,
            M2D_init=M2D_init,
        )
        M2_param = M2_solution.y[0]  # y[0] is M2
        alpha_M = M2_solution.y[1] / M2_param  # y[1] is M2' (dM2/dlna)
        alpha_B_propto = -2 / b * alpha_M  # mochi alpha_B (= - 2* alpha_B_Lombriser)

    alpha_B0 = alpha_B_propto[-1]  # alpha_B0 is the value today (end of lna array)
    print("alpha_B0 from ODE solution:", alpha_B0)
    Delta_Mpl_param = M2_param - 1  # Delta_Mpl is M2 - 1

    ###############################################
    ####### collect stable basis in dict ########
    ###############################################
    stable_basis_dictionary = {
        # background
        "lna_smg": lna_mochi,
        # stable basis
        "D_kin": alpha_param,
        "cs2": cs2_param,
        "Delta_M2": Delta_Mpl_param,
        "alpha_B0": alpha_B0,  # output of mochi
        "alpha_M": alpha_M,
        "w0": w0,
        "wa": wa,
    }

    return stable_basis_dictionary

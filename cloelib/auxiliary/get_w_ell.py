import numpy as np
import pylevin as levin


def get_W_ell(ROOTS, norms, thetagrid, ns, ells, N_thread):
    """
    Precomputes the kernel functions needed to calculate the COSEBI's

    Parameters:
    ROOTS (list): list containing the roots of the cosebi kernel functions, ROOTS[n] should have n + 1 entries
    norms (list): lit containing the the norms, should start at n[1]
    thetagrid (np.array): sufficiently dense grid going from thetamin to thetamax
    ns (list): the indices for the kernel function
    ells (np.array): array with the ells
    N_thread (int): number of threads used to do the integration

    Returns:
    dict with the kernel functions for each n
    """

    w_ell_dict = {}
    for n in ns:
        print(n)
        # Calculate Tn+
        z = np.log(thetagrid / thetagrid[0])
        prod = 1
        for root in ROOTS[n - 1]:
            prod *= z - root
        tp = norms[n] * prod
        # Do the integral to obtain Wn(ell)

        f_of_x = (thetagrid * tp).reshape([len(thetagrid), 1])
        integral_type = 1
        logx = True  # Tells the code to create a logarithmic spline in x for f(x)
        logy = False  # Tells the code to create a logarithmic spline in y for y = f(x)

        lp_single = levin.pylevin(
            integral_type, thetagrid, f_of_x, logx, logy, N_thread
        )

        n_sub = 8  # number of collocation points in each bisection
        n_bisec_max = 32  # maximum number of bisections used
        rel_acc = 1e-8  # relative accuracy target
        boost_bessel = True  # should the bessel functions be calculated with boost instead of GSL, higher accuracy at high Bessel orders
        verbose = False  # should the code talk to you?
        lp_single.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)

        result_levin = np.zeros((len(ells), 1))  # allocate the result
        lp_single.levin_integrate_bessel_single(
            thetagrid[0] * np.ones_like(ells),
            thetagrid[-1] * np.ones_like(ells),
            ells,
            np.zeros_like(ells).astype(int),
            result_levin,
        )

        w_ell_dict[n] = np.array(result_levin)

    return w_ell_dict

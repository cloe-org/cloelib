import numpy as np
import pylevin as levin
import mpmath as mp


def get_roots_and_norms(tmax, tmin, Nmax):
    """
    Calculates the roots,norms and matrix elements given a min and max theta
    up to a certain Nmax.

    Parameters:
    ----------
    tmax: float
        maximum seperation
    tmin: float
        minimum seperation
    Nmax: integer
        maximum COSEBI

    Returns
    -------
    rn (mp.math)
        the roots
    nn (mp.math)
        the normalizations
    coeff_j (mp.math matrix)
        Matrix elements

    Notes
    -----
    Based on the the implementation from OneCovariance by Robert Reischke,
    https://github.com/rreischke/OneCovariance.
    """
    zmax = mp.log(tmax / tmin)
    mp.mp.dps = 150  # decimal precision for mpmath

    # -------------------------
    # coeff_j as mpmath matrix: rows indexed by n (0..Nmax). We'll store full (Nmax+1)x(Nmax+2)
    coeff_j = mp.matrix(Nmax + 1, Nmax + 2)
    # constraint c_n(n+1) = 1 (note: for row index i corresponds to n=i)
    for i in range(Nmax + 1):
        coeff_j[i, i + 1] = mp.mpf(1)

    # --- compute coeffs for n=1 explicitly by solving 2x2 system
    # Build correct mp.matrix for 'aa' and 'bb'
    aa_2x2 = mp.matrix([[J(2, 0, zmax), J(2, 1, zmax)], [J(4, 0, zmax), J(4, 1, zmax)]])
    bb_2x2 = mp.matrix([[-J(2, 2, zmax)], [-J(4, 2, zmax)]])  # since nn=1 so nn+1 = 2

    sol12 = mp.lu_solve(aa_2x2, bb_2x2)  # returns a column matrix (2x1)
    coeff_j[1, 0] = sol12[0, 0]
    coeff_j[1, 1] = sol12[1, 0]

    # -------------------------
    # General solution for nn = 2..Nmax
    for nn in range(2, Nmax + 1):
        size = nn + 1
        aa = mp.matrix(size, size)
        bb = mp.matrix(size, 1)

        # orthogonality conditions
        # there are (nn-1) of these: for m = 1..nn-1 correspond to rows 0..(nn-2)
        for idx_m, m in enumerate(range(1, nn)):
            # fill row idx_m of aa and corresponding entry of bb
            # aa[idx_m, j] = sum_{i=0..m+1} J(1, i+j, zmax) * coeff_j[m, i]
            for j in range(0, nn + 1):
                s = mp.mpf(0)
                for i in range(0, m + 2):
                    s += J(1, i + j, zmax) * coeff_j[m, i]
                aa[idx_m, j] = s
            # RHS
            rhs = mp.mpf(0)
            for i in range(0, m + 2):
                rhs -= J(1, i + nn + 1, zmax) * coeff_j[m, i]
            bb[idx_m, 0] = rhs

        for j in range(nn + 1):
            aa[nn - 1, j] = J(2, j, zmax)
            aa[nn, j] = J(4, j, zmax)
        bb[nn - 1, 0] = -J(2, nn + 1, zmax)
        bb[nn, 0] = -J(4, nn + 1, zmax)

        # solve
        sol = mp.lu_solve(aa, bb)  # size x 1
        # place into coeff_j row 'nn'
        for j in range(size):
            coeff_j[nn, j] = sol[j, 0]

    coeff_j = coeff_j[1:, :]  # now row 0 corresponds to n=1

    # -------------------------
    # Normalizations Nn
    Nn = []
    for nn in range(1, Nmax + 1):
        temp_sum = mp.mpf(0)
        for i in range(nn + 2):
            for j in range(nn + 2):
                temp_sum += coeff_j[nn - 1, i] * coeff_j[nn - 1, j] * J(1, i + j, zmax)
        temp_Nn = (mp.expm1(zmax)) / temp_sum
        temp_Nn = mp.sqrt(mp.fabs(temp_Nn))
        Nn.append(temp_Nn)

    ##We now want the root of the filter t_+n^log
    # the filter is:
    rn = []
    for nn in range(1, Nmax + 1):
        rn.append(
            mp.polyroots(coeff_j[nn - 1, : nn + 2][::-1], maxsteps=500, extraprec=100)
        )
        # [::-1])

    # -----------------
    return rn, Nn, coeff_j


def J(k, j, zmax):
    """Helper function for get_roots to calculate a gamma function"""

    # J = (Gamma(j+1) - gamma_upper(j+1, -k zmax)) / (-k)^(j+1)
    # Use mpmath routines with high precision
    gamma_full = mp.gamma(j + 1)
    gamma_upper = mp.gammainc(j + 1, -k * zmax)  # corresponds to upper incomplete?
    numerator = gamma_full - gamma_upper
    denom = mp.power(-k, j + 1)
    return mp.fdiv(numerator, denom)


def tp(n, z, nn, rn):
    """
    kernel function Tn+

    Parameters:
    ----------
    n: integer
        cosebi index
    z: float or array
        log(theta/theta_min)
    nn: mpmath
        normalizations from get_roots
    rn: mpmath
        roots from get_roots

    Returns
    -------
    tn+ (mp.math)
        the kernel function tn+

    """
    prod = mp.mpf(1)
    for root in rn[n - 1]:
        prod *= z - root
    return nn[n - 1] * prod


def an2(n, nn, coeff_j):
    """Helper function for tm"""
    s = mp.mpf(0)
    for j in range(0, n + 2):  # j = 0 .. n+1
        term = nn[n - 1] * coeff_j[(n - 1, j)] * mp.factorial(j) / ((-2) ** (j + 1))
        s += term
    return 4 * s


def an4(n, nn, coeff_j):
    """Helper function for tm"""
    s = mp.mpf(0)
    for j in range(0, n + 2):
        term = nn[n - 1] * coeff_j[(n - 1, j)] * mp.factorial(j) / ((-4) ** (j + 1))
        s += term
    return 12 * s


def dnm(n, m, nn, coeff_j):
    """Helper function for tm"""
    s = mp.mpf(0)
    for j in range(m, n + 2):  # j = m .. n+1
        power_term = (-2) ** (m - j - 1)
        bracket = 3 * (2 ** (m - j - 1)) - 1
        term = nn[n - 1] * coeff_j[(n - 1, j)] * mp.factorial(j) * power_term * bracket
        s += term
    return nn[n - 1] * coeff_j[(n - 1, m)] + (4 / mp.factorial(m)) * s


def tm(n, z, nn, coeff_j):
    """
    kernel function Tm-

    Parameters:
    ----------
    n: integer
        cosebi index
    z: float or array
        log(theta/theta_min)
    nn: mpmath
        normalizations from get_roots
    rn: coeff_j
        matrix elements from get_roots

    Returns
    -------
    tm (mp.math)
        tminus kernel function

    """
    s = mp.mpf(0)
    for m in range(0, n + 1):
        s += dnm(n, m, nn, coeff_j) * (z**m)
    return (
        an2(n, nn, coeff_j) * mp.e ** (-2 * z)
        - an4(n, nn, coeff_j) * mp.e ** (-4 * z)
        + s
    )


def get_W_ell(thetagrid, Nmax, ells, N_thread):
    """
    kernel function Tm-

    Parameters:
    ----------
    n: integer
        cosebi index
    z: float or array
        log(theta/theta_min)
    nn: mpmath
        normalizations from get_roots
    rn: coeff_j
        matrix elements from get_roots

    Returns
    -------
    tm (mp.math)
        tminus kernel function
    """

    print("start calculating roots and norms:")

    tmax = thetagrid[-1]
    tmin = thetagrid[0]

    rn, nn, coeff_j = get_roots_and_norms(tmax, tmin, Nmax)
    print("done")
    ns = np.arange(1, Nmax)
    w_ells = {}

    print("start performing the bessel integrals")
    for n in ns:
        print(n, "/", Nmax)
        Tm = tm(n, np.log(thetagrid / thetagrid[0]), nn, coeff_j)
        Tm = np.array([complex(x).real for x in Tm])
        f_of_x = (thetagrid * Tm).reshape([len(thetagrid), 1])

        integral_type = 1
        logx = True  # Tells the code to create a logarithmic spline in x for f(x)
        logy = True  # Tells the code to create a logarithmic spline in y for y = f(x)

        lp_single = levin.pylevin(
            integral_type, thetagrid, f_of_x, logx, logy, N_thread
        )

        n_sub = 32  # number of collocation points in each bisection
        n_bisec_max = 8  # maximum number of bisections used
        rel_acc = 1e-8  # relative accuracy target
        boost_bessel = True  # should the bessel functions be calculated with boost instead of GSL, higher accuracy at high Bessel orders
        verbose = False  # should the code talk to you?
        lp_single.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)

        result_levin = np.zeros((len(ells), 1))  # allocate the result
        lp_single.levin_integrate_bessel_single(
            thetagrid[0] * np.ones_like(ells),
            thetagrid[-1] * np.ones_like(ells),
            ells,
            4 * np.ones_like(ells).astype(int),
            result_levin,
        )

        w_ells[n] = result_levin
    return w_ells

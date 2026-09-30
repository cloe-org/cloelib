import functools
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
    mp.mp.dps = 150  # decimal precision for mpmath, set before any mpmath value
    zmax = mp.log(mp.mpf(tmax) / mp.mpf(tmin))
    # J(k, j, zmax) is called repeatedly with the same arguments below
    Jc = functools.lru_cache(maxsize=None)(J)

    # -------------------------
    # coeff_j as mpmath matrix: rows indexed by n (0..Nmax). We'll store full (Nmax+1)x(Nmax+2)
    coeff_j = mp.matrix(Nmax + 1, Nmax + 2)
    # constraint c_n(n+1) = 1 (note: for row index i corresponds to n=i)
    for i in range(Nmax + 1):
        coeff_j[i, i + 1] = mp.mpf(1)

    # --- compute coeffs for n=1 explicitly by solving 2x2 system
    # Build correct mp.matrix for 'aa' and 'bb'
    aa_2x2 = mp.matrix(
        [[Jc(2, 0, zmax), Jc(2, 1, zmax)], [Jc(4, 0, zmax), Jc(4, 1, zmax)]]
    )
    bb_2x2 = mp.matrix([[-Jc(2, 2, zmax)], [-Jc(4, 2, zmax)]])  # since nn=1 so nn+1 = 2

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
                    s += Jc(1, i + j, zmax) * coeff_j[m, i]
                aa[idx_m, j] = s
            # RHS
            rhs = mp.mpf(0)
            for i in range(0, m + 2):
                rhs -= Jc(1, i + nn + 1, zmax) * coeff_j[m, i]
            bb[idx_m, 0] = rhs

        for j in range(nn + 1):
            aa[nn - 1, j] = Jc(2, j, zmax)
            aa[nn, j] = Jc(4, j, zmax)
        bb[nn - 1, 0] = -Jc(2, nn + 1, zmax)
        bb[nn, 0] = -Jc(4, nn + 1, zmax)

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
                temp_sum += coeff_j[nn - 1, i] * coeff_j[nn - 1, j] * Jc(1, i + j, zmax)
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

    # -----------------
    return rn, Nn, coeff_j


def J(k, j, zmax):
    """Helper function for get_roots to calculate a gamma function"""
    # using lower gamma (J = mp.gammainc(j+1,0,-k*zmax)) function gives an error, so we go via the upper
    # J = (Gamma(j+1) - gamma_upper(j+1, -k zmax)) / (-k)^(j+1)
    # Use mpmath routines with high precision

    gamma_full = mp.gamma(j + 1)
    gamma_upper = mp.gammainc(j + 1, -k * zmax)
    numerator = gamma_full - gamma_upper
    denom = mp.power(-k, j + 1)
    return mp.fdiv(numerator, denom)


def tp(n, t, tmin, nn, rn):
    """
    kernel function Tn+

    Parameters:
    ----------
    n: integer
        cosebi index
    t:  np.array or list
        theta, angular seperation
    tmax: float
        max angular seperation
    nn: mpmath
        normalizations from get_roots
    rn: mpmath
        roots from get_roots

    Returns
    -------
    tn+ (mp.math)
        the kernel function tn+

    """

    # np.array to allow for simple multiplication
    z = np.array([mp.log(x / tmin) for x in t])
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


def tm(n, t, tmin, nn, coeff_j):
    """
    kernel function Tm-

    Parameters:
    ----------
    n: integer
        cosebi index
    t: np.array or list
        theta, angular seperation
    tmin: float
        min angular seperation
    nn: mpmath
        normalizations from get_roots
    coeff_j:
        matrix elements from get_roots

    Returns
    -------
    tm (mp.math)
        tminus kernel function

    """

    # np.array to allow for simple multiplication
    z = np.array([mp.log(x / tmin) for x in t])
    s = mp.mpf(0)
    for m in range(0, n + 1):
        s += dnm(n, m, nn, coeff_j) * (z**m)
    return (
        an2(n, nn, coeff_j) * mp.e ** (-2 * z)
        - an4(n, nn, coeff_j) * mp.e ** (-4 * z)
        + s
    )


def _chebyshev_fit(func, zmax, tol=mp.mpf("1e-17"), deg=64, max_deg=1024):
    """
    Chebyshev interpolant of an mpmath function on ``z`` in ``[0, zmax]``.

    The function is sampled at Chebyshev nodes in mpmath precision and the
    coefficients are computed in mpmath, doubling the degree until the tail
    coefficients drop below ``tol`` relative to the largest one. Only the
    final coefficients are cast to float64, so evaluating the interpolant in
    float64 (Clenshaw recurrence) does not suffer from the cancellation of the
    power-basis polynomial.

    Returns
    -------
    np.ndarray
        Chebyshev coefficients (float64) in the variable ``x = 2 z / zmax - 1``.
    """
    zmax = mp.mpf(zmax)
    while True:
        N = deg + 1
        angles = [mp.pi * (k + mp.mpf(1) / 2) / N for k in range(N)]
        vals = [func((mp.cos(a) + 1) * zmax / 2) for a in angles]
        coeffs = [
            2 * mp.fsum(v * mp.cos(j * a) for v, a in zip(vals, angles)) / N
            for j in range(N)
        ]
        coeffs[0] /= 2
        scale = max(mp.fabs(c) for c in coeffs)
        if max(mp.fabs(c) for c in coeffs[-8:]) <= tol * scale or deg >= max_deg:
            return np.array([float(c) for c in coeffs])
        deg *= 2


def _eval_chebyshev(coeffs, thetagrid, tmin, zmax):
    """Evaluate a Chebyshev series from `_chebyshev_fit` at z = log(theta / tmin)."""
    x = 2 * np.log(thetagrid / tmin) / float(zmax) - 1
    return np.polynomial.chebyshev.chebval(x, coeffs)


def _tm_fast(n, thetagrid, tmin, nn, coeff_j):
    """
    Fast evaluation of T_n^-(theta) over the full thetagrid.

    T_n^- (power-law terms and log-polynomial together) is expanded in a
    Chebyshev series in z = log(theta / tmin) computed at mpmath precision
    (see ``_chebyshev_fit``), then evaluated in float64. This avoids the
    catastrophic cancellation of the power-basis log-polynomial (z ~ 6.4 for
    th_max/th_min = 600) at the cost of a few hundred mpmath evaluations per
    mode instead of one per theta point.
    """
    zmax = mp.log(mp.mpf(thetagrid[-1]) / mp.mpf(tmin))
    a2 = an2(n, nn, coeff_j)
    a4 = an4(n, nn, coeff_j)
    dnm_coeffs_highfirst = [dnm(n, m, nn, coeff_j) for m in range(n, -1, -1)]

    def tm_z(z):
        return (
            a2 * mp.exp(-2 * z)
            - a4 * mp.exp(-4 * z)
            + mp.polyval(dnm_coeffs_highfirst, z)
        )

    return _eval_chebyshev(_chebyshev_fit(tm_z, zmax), thetagrid, tmin, zmax)


def _tp_fast(n, thetagrid, tmin, nn, coeff_j):
    """
    Fast evaluation of T_n^+(theta) = N_n sum_j c_nj z^j over the full thetagrid,
    with z = log(theta / tmin), using the same Chebyshev approach as ``_tm_fast``.
    """
    zmax = mp.log(mp.mpf(thetagrid[-1]) / mp.mpf(tmin))
    coeffs_highfirst = [nn[n - 1] * coeff_j[n - 1, j] for j in range(n + 1, -1, -1)]

    def tp_z(z):
        return mp.polyval(coeffs_highfirst, z)

    return _eval_chebyshev(_chebyshev_fit(tp_z, zmax), thetagrid, tmin, zmax)


def get_T_plus_minus(thetagrid, Nmax):
    """
    Real-space COSEBI kernels T_n^+(theta) and T_n^-(theta) for n = 1..Nmax.

    The output layout is the one expected by
    ``cloelib.summary_statistics.angular_two_point.get_cosebis_from_2pcf``.

    Parameters
    ----------
    thetagrid : np.ndarray
        Angular separation grid in radians; its first and last values set the
        COSEBI range ``[theta_min, theta_max]``.
    Nmax : int
        Maximum COSEBI mode index.

    Returns
    -------
    T_plus, T_minus : np.ndarray
        Arrays of shape ``(Nmax + 1, len(thetagrid))``; row ``n`` holds mode
        ``n`` and row 0 is zero, so ``T_plus[ns]`` selects modes ``ns``.
    """
    tmin, tmax = thetagrid[0], thetagrid[-1]
    rn, nn, coeff_j = get_roots_and_norms(tmax, tmin, Nmax)
    T_plus = np.zeros((Nmax + 1, len(thetagrid)))
    T_minus = np.zeros_like(T_plus)
    for n in range(1, Nmax + 1):
        T_plus[n] = _tp_fast(n, thetagrid, tmin, nn, coeff_j)
        T_minus[n] = _tm_fast(n, thetagrid, tmin, nn, coeff_j)
    return T_plus, T_minus


def get_W_ell(thetagrid, Nmax, ells, N_thread):
    """
    Compute harmonic-space COSEBI kernels W_n(ell) for n = 1..Nmax.

    The T_n^- real-space kernels are expanded in Chebyshev series computed at
    mpmath precision (150 decimal digits) and evaluated in float64 via
    ``_tm_fast``, so their cost does not grow with the size of `thetagrid`.
    All Nmax kernels are then stacked into a single ``(N_theta, Nmax)``
    integrand matrix and passed to pylevin in **one batched Levin call**.

    The Levin integration dominates the run time and scales linearly with
    ``len(ells)``. Since the COSEBI integrals account for the grid spacing
    (see ``cloelib.auxiliary.math_utils.quadrature_weights``), `ells` does not
    need to be unit-spaced: a uniform grid only has to resolve the oscillations
    of W_n(ell), whose period is roughly ``2 pi / theta_max``. For example,
    for theta in [0.5, 300] arcmin, ``np.arange(2, 60000, 5)`` reproduces the
    unit-spaced result to ~1e-5 at one fifth of the cost.

    Parameters
    ----------
    thetagrid : np.ndarray
        Angular separation grid in radians (log-spaced).
    Nmax : int
        Maximum COSEBI mode index.
    ells : np.ndarray
        Multipoles at which W_n(ell) is evaluated. Pass the same grid to
        ``get_cosebis_from_cl``/``AngularTwoPoint.get_cosebis``.
    N_thread : int
        Number of threads passed to pylevin.

    Returns
    -------
    dict
        Keys ``1..Nmax`` map to 1-D arrays of length ``len(ells)``;
        key ``"metadata"`` holds ``{"THMIN": tmin, "THMAX": tmax}`` in radians
        (the unit of `thetagrid`). The COSEBI outputs built from these kernels
        store ``thmin``/``thmax`` in arcmin.
    """
    print("start calculating roots and norms:")

    tmax = thetagrid[-1]
    tmin = thetagrid[0]

    rn, nn, coeff_j = get_roots_and_norms(tmax, tmin, Nmax)
    print("done")
    ns = np.arange(1, Nmax + 1)

    # ------------------------------------------------------------------
    # Step 1: evaluate all T_n^- kernels (full mpmath precision, no change)
    # Shape of f_of_x: (N_theta, Nmax) — each column is theta * T_n^-(theta)
    # ------------------------------------------------------------------
    print("start evaluating kernels")
    f_of_x = np.column_stack(
        [thetagrid * _tm_fast(n, thetagrid, tmin, nn, coeff_j) for n in ns]
    )  # (N_theta, Nmax)

    # ------------------------------------------------------------------
    # Step 2: single batched Levin call over all Nmax columns at once.
    # pylevin treats each column of f_of_x as a separate integrand and
    # returns result_levin of shape (len(ells), Nmax) in one C++ pass,
    # saving Nmax-1 Python-level object creations and Levin setups.
    # ------------------------------------------------------------------
    print("start performing the bessel integrals (batched over all modes)")
    integral_type = 1
    logx = True  # logarithmic spline in x
    logy = True  # logarithmic spline in y (T_n^- is positive on [tmin, tmax])

    lp_all = levin.pylevin(integral_type, thetagrid, f_of_x, logx, logy, N_thread)

    n_sub = 32  # collocation points per bisection
    n_bisec_max = 8  # maximum bisections
    rel_acc = 1e-8  # relative accuracy target
    boost_bessel = True  # use Boost Bessel functions (higher accuracy)
    verbose = False
    lp_all.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)

    result_levin = np.zeros((len(ells), Nmax))
    lp_all.levin_integrate_bessel_single(
        thetagrid[0] * np.ones_like(ells),
        thetagrid[-1] * np.ones_like(ells),
        ells,
        4 * np.ones_like(ells).astype(int),
        result_levin,
    )

    # Unpack columns back into the expected {n: array} dict layout
    w_ells = {n: result_levin[:, n - 1] for n in ns}
    w_ells["metadata"] = {
        "THMIN": tmin,
        "THMAX": tmax,
    }
    return w_ells

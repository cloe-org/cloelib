import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import jit
from scipy.special import hyp2f1
from scipy.integrate import quad


def get_z_bin_jax(z, zbin_edges):
    """JAX-compatible bin index for use inside jit/odeint."""
    zbin_edges = jnp.asarray(zbin_edges)
    bin_idx = jnp.digitize(z, zbin_edges, right=False) - 1
    last = zbin_edges.shape[0] - 2
    bin_idx = jnp.clip(bin_idx, 0, last)
    bin_idx = jnp.where(z >= zbin_edges[-1], last, bin_idx)
    return bin_idx

@jit
def interp1d_jax(x, xp, fp):
    """
    JAX-compatible 1D linear interpolation function.

    Args:
        x   (array): points at which to interpolate
        xp  (array): 1D array of x-coordinates (must be sorted)
        fp  (array): 1D array of y-coordinates corresponding to xp

    Returns:
        array: interpolated values at x
    """
    n = xp.shape[0]
    indices = jnp.clip(jnp.searchsorted(xp, x) - 1, 0, n - 2)
    x0 = xp[indices]
    x1 = xp[indices + 1]
    y0 = fp[indices]
    y1 = fp[indices + 1]
    dx = x1 - x0
    # Avoid division by zero (shouldn't happen for sorted arrays, but safety check)
    dx_safe = jnp.where(dx == 0, 1.0, dx)
    return y0 + (y1 - y0) * (x - x0) / dx_safe


@jit
def trapz_jax(y, x):
    """
    JAX-compatible trapezoidal integration.

    Args:
        y   (array): function values
        x   (array): integration points (must be sorted)

    Returns:
        scalar: integral value
    """
    dx = x[1:] - x[:-1]
    return jnp.sum(0.5 * dx * (y[1:] + y[:-1]))


@jit
def cumulative_integral(y, x):
    """
    JAX-compatible cumulative trapezoidal integration.

    Args:
        y   (array): function values
        x   (array): integration points (must be sorted)

    Returns:
        array: cumulative integral values (same length as x, first value is 0)
    """
    dx = x[1:] - x[:-1]
    trapz_contrib = 0.5 * dx * (y[:-1] + y[1:])
    cumsum = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(trapz_contrib)])
    return cumsum


@jit
def compute_cumulative_integral(a_fine, a_grid, w_vals):
    """Pre-compute cumulative integral from a_fine to 1.0 for all a_fine values.

    Computes I(a) = integral from a to 1.0 of 3*(1+w(a'))/a' da'
    """
    w_fine = interp1d_jax(a_fine, a_grid, w_vals)
    integrand = 3.0 * (1.0 + w_fine) / a_fine

    # Compute cumulative integral backwards from 1.0 to a_start using cumsum
    # Reverse arrays, compute forward cumulative sum, then reverse back
    a_rev = a_fine[::-1]
    integrand_rev = integrand[::-1]

    # Compute differences and trapezoidal rule contributions
    da_rev = a_rev[:-1] - a_rev[1:]  # Note: reversed, so this is positive
    # Trapezoidal contributions: 0.5 * da * (f[i] + f[i+1])
    # In reversed order: f[i+1] is at index i+1 in reversed array
    trapz_contrib = 0.5 * da_rev * (integrand_rev[:-1] + integrand_rev[1:])

    # Cumulative sum backwards (from 1.0 to a_start)
    # Start with 0 at a=1.0, accumulate backwards
    cumsum_rev = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(trapz_contrib)])

    # Reverse back to get forward order (from a_start to 1.0)
    integral = cumsum_rev[::-1]

    return integral


@jit
def Omega_m_DE_fast(a, omega0, integral_interp, a_fine):
    """Fast version using pre-computed cumulative integral."""
    integral = interp1d_jax(a, a_fine, integral_interp)
    omegaL = (1.0 - omega0) * jnp.exp(integral)
    E2 = omega0 / a**3 + omegaL
    return omega0 / a**3 / E2


@jit
def Omega_m_DE(a, omega0, a_grid, w_vals, n_points=128):
    """Original version - kept for backward compatibility if needed.

    Args:
        n_points: Number of integration points (default 128, reduced from 256 for speed)
    """

    def integrand(a_in):
        w = interp1d_jax(a_in, a_grid, w_vals)
        return 3.0 * (1.0 + w) / a_in

    a_eval = jnp.linspace(a, 1.0, n_points)
    w_integrand = integrand(a_eval)
    integral = trapz_jax(w_integrand, a_eval)

    omegaL = (1.0 - omega0) * jnp.exp(integral)
    E2 = omega0 / a**3 + omegaL
    return omega0 / a**3 / E2


@jit
def Omega_m(a, omega0, w0, wa):
    """
    Computes a function :math:`\Omega_\mathrm{m}(a)` as a function of the scale-factor:

    .. math::
          \Omega_\mathcal{m}(a) = \\frac{\Omega_{m,0} a^{-3}}
          {E^2(a)} = \\frac{\Omega_{m,0} a^{-3}}
          {\Omega_{m,0} a^{-3}+\Omega_\mathrm{DE}(a)}
    assuming a flat universe with the expansion history :math:`H(a)=H_0 E(a)` the dark
    energy component that evolves as
    :math:`\Omega_\mathrm{DE}(a) = (1-\Omega_\mathrm{m, 0}) a^{-3(1+w_0+w_a)} \mathrm{e}^{3(a-1)w_a}`.

    Args:
        a       (array):  scale factor array, is strictly increasing

        omega0  (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

        w0      (scalar): dark energy equation of state parameter

        wa      (scalar): dark energy equation of state parameter

    Returns:
        array: Omega_m(a)
    """
    omegaL = (
        (1.0 - omega0) * a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(3.0 * (-1.0 + a) * wa)
    )
    E2 = omega0 / a**3 + omegaL
    return omega0 / a**3 / E2


@jit
def dlnH_dlna_DE_fast(a, omega0, integral_interp, a_fine, a_grid, w_vals):
    """Fast version using pre-computed cumulative integral."""
    integral = interp1d_jax(a, a_fine, integral_interp)
    omegaL = (1.0 - omega0) * jnp.exp(integral)
    E2 = omega0 / a**3 + omegaL
    w_a = interp1d_jax(a, a_grid, w_vals)
    return -1.5 * (omega0 / a**3 + (1 + w_a) * omegaL) / E2


@jit
def dlnH_dlna_DE(a, omega0, a_grid, w_vals, n_points=128):
    """Original version - kept for backward compatibility if needed.

    Args:
        n_points: Number of integration points (default 128, reduced from 256 for speed)
    """

    def integrand(a_in):
        w = interp1d_jax(a_in, a_grid, w_vals)
        return 3.0 * (1.0 + w) / a_in

    a_eval = jnp.linspace(a, 1.0, n_points)
    w_integrand = integrand(a_eval)
    integral = trapz_jax(w_integrand, a_eval)

    omegaL = (1.0 - omega0) * jnp.exp(integral)
    E2 = omega0 / a**3 + omegaL
    w_a = interp1d_jax(a, a_grid, w_vals)
    return -1.5 * (omega0 / a**3 + (1 + w_a) * omegaL) / E2


@jit
def dlnH_dlna(a, omega0, w0, wa):
    """
    Computes the derivative of the ln(expansion) with respect to ln(scale factor):

    .. math::
          \\frac{\mathrm{d} \ln{H}}{\mathrm{d} \ln{a}} =
           -\\frac{3}{2} \\frac{\Omega_\mathrm{m} a^{-3} + (1+w_0+w_a[1-a])\Omega_\mathrm{DE}(a)}
          {\Omega_{m,0} a^{-3}+\Omega_\mathrm{DE}(a)}

    Args:
        a       (array):  scale factor array, is strictly increasing

        omega0  (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

        w0      (scalar): dark energy equation of state parameter

        wa      (scalar): dark energy equation of state parameter

    Returns:
        array: dlnH_dlna(a)
    """
    omegaL = (
        (1.0 - omega0) * a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(3.0 * (-1.0 + a) * wa)
    )
    E2 = omega0 / a**3 + omegaL
    return -1.5 * (omega0 / a**3 + (1 + w0 + wa * (1.0 - a)) * omegaL) / E2


@jit
def DE_D_derivatives_fast(D, a, integral_interp, a_fine, a_grid, w_vals, omega0):
    """Fast version using pre-computed cumulative integral."""
    D1, D2 = D
    dlnH = dlnH_dlna_DE_fast(a, omega0, integral_interp, a_fine, a_grid, w_vals)
    Om = Omega_m_DE_fast(a, omega0, integral_interp, a_fine)
    dD1 = D2
    dD2 = -D2 / a * (3.0 + dlnH) + 1.5 * D1 / a**2 * Om
    return jnp.array([dD1, dD2])


@jit
def DE_D_derivatives(D, a, a_grid, w_vals, omega0):
    """Original version - kept for backward compatibility if needed."""
    D1, D2 = D
    dlnH = dlnH_dlna_DE(a, omega0, a_grid, w_vals)
    Om = Omega_m_DE(a, omega0, a_grid, w_vals)
    dD1 = D2
    dD2 = -D2 / a * (3.0 + dlnH) + 1.5 * D1 / a**2 * Om
    return jnp.array([dD1, dD2])


@jit
def DE_fde_D_derivatives(D, a, fde_vals, omega0, zbin_edges):
    """Original version - kept for backward compatibility if needed."""
    D1, D2 = D
    omegaL = (1.0 - omega0) * fde_vals[get_z_bin_jax(1.0 / a - 1.0, zbin_edges)]
    E2 = omega0 / a**3 + omegaL
    # f'=0 inside bins ⇒ like w_eff = -1
    dlnH = -1.5 * (omega0 / a**3) / E2
    Om = (omega0 / a**3) / E2
    dD1 = D2
    dD2 = -D2 / a * (3.0 + dlnH) + 1.5 * D1 / a**2 * Om
    return jnp.array([dD1, dD2])

@jit
def D_derivatives_LCDM(D, a, omega0):
    """
    Highly optimized version for LCDM (w0=-1, wa=0) that is used with jax.experimental.ode.odeint
    to solve the growth factor differential equation. This version eliminates redundant computations
    by using the fact that dark energy is constant for LCDM.

    Args:
        D       (array): growth factor, is strictly increasing

        a       (array):  scale factor array

        omega0  (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

    Returns:
        array: D(a)
    """
    D1, D2 = D
    # For LCDM: omegaL = (1-omega0) (constant dark energy)
    omega_m_a = omega0 / (a**3)
    omegaL = 1.0 - omega0
    E2 = omega_m_a + omegaL
    # Simplified dlnH_dlna for LCDM
    dlnH = -1.5 * omega_m_a / E2
    Om = omega_m_a / E2

    dD1 = D2
    dD2 = -D2 / a * (3.0 + dlnH) + 1.5 * D1 / a**2 * Om
    return jnp.array([dD1, dD2])


@jit
def D_derivatives(D, a, omega0, w0, wa):
    """
    Function that is used with jax.experimental.ode.odeint to solve the following differential
    equation for the growth factor :math:`D(a)`:

    .. math::
        \ddot{D} +\left( 3 + \\frac{\mathrm{d} \ln{H}}{\mathrm{d} \ln{a}} \\right) \\frac{\dot{D}}{a}-\\frac{3}{2} \Omega_\mathrm{m}(a) \\frac{D}{a^2} = 0

    with :math:`\dot{}` denoting a derivative with respect to the scale factor.
    Optimized version that combines computations from dlnH_dlna and Omega_m.

    Args:
        D       (array): growth factor, is strictly increasing

        a       (array):  scale factor array

        omega0  (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

        w0      (scalar): dark energy equation of state parameter

        wa      (scalar): dark energy equation of state parameter

    Returns:
        array: D(a)
    """
    D1, D2 = D
    # Compute common terms once
    a3 = a**3
    omega_m_a = omega0 / a3
    omegaL = (
        (1.0 - omega0) * a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(3.0 * (-1.0 + a) * wa)
    )
    E2 = omega_m_a + omegaL

    # Compute dlnH_dlna and Omega_m from shared terms
    dlnH = -1.5 * (omega_m_a + (1 + w0 + wa * (1.0 - a)) * omegaL) / E2
    Om = omega_m_a / E2

    # Growth equation derivatives
    dD1 = D2
    dD2 = -D2 / a * (3.0 + dlnH) + 1.5 * D1 / a**2 * Om
    return jnp.array([dD1, dD2])


@jit
def Friction(a, omega0, h, w0, wa, xi):
    """
    Computes the modification to Euler equation in the interacting dark energy scenario.

    Args:
        a       (array):  scale factor array, is strictly increasing

        omega0  (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

        h       (scalar): value of the Hubble constant as in :math:`H_0 = 100 h/` Mpc

        w0      (scalar): dark energy equation of state parameter

        wa      (scalar): dark energy equation of state parameter

        xi      (scalar): scattering strength

    Returns:
        array: Friction(a)
    """
    c3 = 0.09747163203504212
    omegaL = (
        (1.0 - omega0) * a ** (-3.0 * (1.0 + w0 + wa)) * jnp.exp(3.0 * (-1.0 + a) * wa)
    )
    E = jnp.sqrt(omega0 / a**3 + omegaL)
    Fr = (1 + w0 + wa * (1.0 - a)) * xi * omegaL * h * c3 / E
    return Fr


@jit
def IDE_D_derivatives(D, a, omega0=0.3, h=0.68, w0=-1, wa=0, xi=0):
    """
    Function that is used with jax.experimental.ode.odeint to solve the following differential
    equation for the modified growth factor :math:`D(a)` in interacting dark energy:

    .. math::
        \ddot{D} +\left( 3 + \mathrm{Friction} + \\frac{\mathrm{d} \ln{H}}{\mathrm{d} \ln{a}} \\right) \\frac{\dot{D}}{a}-\\frac{3}{2} \Omega_\mathrm{m}(a) \\frac{D}{a^2} = 0

    with :math:`\dot{}` denoting a derivative with respect to the scale factor and
    :math:`\mathrm{Friction} = (1+w(a))\\xi \\frac{\Omega_\mathrm{DE}(a) \\rho_\mathrm{crit}}{H(a)}` being a modification to the Euler equation.

    Args:
        D           (array): growth factor

        a           (array):  scale factor array, is strictly increasing

        omega0      (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

        h           (scalar): value of the Hubble constant as in :math:`H_0 = 100 h/` Mpc

        w0          (scalar): dark energy equation of state parameter

        wa          (scalar): dark energy equation of state parameter

        xi          (scalar): scattering strength

    Returns:
        array: D(a)
    """
    D1, D2 = D
    dlnH = dlnH_dlna(a, omega0, w0, wa)
    Om = Omega_m(a, omega0, w0, wa)
    dD1 = D2
    dD2 = (
        -D2 / a * (3.0 + Friction(a, omega0, h, w0, wa, xi) + dlnH)
        + 1.5 * D1 / a**2 * Om
    )
    return jnp.array([dD1, dD2])


@jit
def MG_D_derivatives(D, a, a_grid, mu_vals, omega0=0.3, w0=-1, wa=0):
    """
    Function that is used with jax.experimental.ode.odeint to solve the following differential
    equation for the modified growth factor :math:`D(a)`:

    .. math::
        \ddot{D} +\left( 3 + \\frac{\mathrm{d} \ln{H}}{\mathrm{d} \ln{a}} \\right) \\frac{\dot{D}}{a}-\\frac{3}{2} \mu(a) \Omega_\mathrm{m}(a) \\frac{D}{a^2}= 0

    with :math:`\dot{}` denoting a derivative with respect to the scale factor and
    :math:`\mu = \\frac{G_\mathrm{eff}(a)}{G}` being a modification to the gravitational constant.

    Args:
        D           (array): growth factor

        a           (array):  scale factor array, is strictly increasing

        a_grid      (array): scale factor grid for :math:`\mu(a)` interpolation

        mu_vals     (array): :math:`\mu(a)` values on a_grid

        omega0      (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

        w0          (scalar): dark energy equation of state parameter

        wa          (scalar): dark energy equation of state parameter

    Returns:
        array: D(a)
    """
    D1, D2 = D
    dlnH = dlnH_dlna(a, omega0, w0, wa)
    Om = Omega_m(a, omega0, w0, wa)
    mu_a = interp1d_jax(a, a_grid, mu_vals)
    dD1 = D2
    dD2 = -D2 / a * (3.0 + dlnH) + 1.5 * mu_a * D1 / a**2 * Om
    return jnp.array([dD1, dD2])


class MGrowth(object):
    def __init__(self, CosmoDict):
        """
        Minimal initialization requires the definition of a dictionary including
        only the cosmological parameters that are necessary to compute the expansion
        history and the scale factor(s).
        If the cosmology is not specified, hardcoded intial values are assumed.
        """

        if CosmoDict is None:
            self.omega0 = 0.3
            self.h = 0.68
            self.w0 = -1.0
            self.wa = 0.0
            self.a_arr = [1.0]
        else:
            self.omega0 = CosmoDict["Omega_m"]
            self.h = CosmoDict["h"]
            self.w0 = CosmoDict["w0"] if "w0" in CosmoDict else -1.0
            self.wa = CosmoDict["wa"] if "wa" in CosmoDict else 0.0
            self.a_arr = CosmoDict["a_arr"]

        self.a_start = 1.0e-4
        self.aa = jnp.concatenate([jnp.array([self.a_start]), jnp.array(self.a_arr)])
        self.aa_interp = jnp.logspace(-5, 1.5, 512)


class w_a(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def growth_parameters(self, a_grid, w_vals, use_fast=True, n_fine=512):
        """
        Computes growth parameters using optimized pre-computed integrals.

        Args:
            a_grid: Scale factor grid for w(a) interpolation
            w_vals: w(a) values on a_grid
            use_fast: If True, use pre-computed integrals (much faster). If False, use original method.
            n_fine: Number of points for fine integration grid (only used if use_fast=True)
        """
        if use_fast:
            # Create fine grid for pre-computing integrals
            a_fine = jnp.linspace(self.a_start, 1.0, n_fine)
            # Pre-compute cumulative integral once
            integral_interp = compute_cumulative_integral(a_fine, a_grid, w_vals)
            # Use fast version with pre-computed integrals
            D_sol = odeint(
                DE_D_derivatives_fast,
                jnp.array([self.a_start, 1.0]),
                self.aa,
                integral_interp,
                a_fine,
                a_grid,
                w_vals,
                self.omega0,
            )
        else:
            # Original method (slower but kept for comparison)
            D_sol = odeint(
                DE_D_derivatives,
                jnp.array([self.a_start, 1.0]),
                self.aa,
                a_grid,
                w_vals,
                self.omega0,
            )
        D, dDda = D_sol.T
        f = self.aa * dDda / D
        return D[1:], f[1:]

class fde_a(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def growth_parameters(self, fde_vals, zbin_edges):
        """
        Computes growth for piecewise-constant f_DE(z) (constant within redshift bins).

        Args:
            fde_vals: per-bin f_DE amplitudes, length = len(zbin_edges) - 1
            zbin_edges: redshift bin edges (ascending)
        """
        D_sol = odeint(
            DE_fde_D_derivatives,
            jnp.array([self.a_start, 1.0]),
            self.aa,
            fde_vals,
            self.omega0,
            zbin_edges,
        )
        D, dDda = D_sol.T
        f = self.aa * dDda / D
        return D[1:], f[1:]

class mu_a(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def growth_parameters(self, a_grid, mu_vals):
        """
        Computes growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation for a modified cosmology with custom :math:`\mu(a)`.

        Args:
            a_grid     (array): scale factor grid for :math:`\mu(a)` interpolation, should
                        allow for :math:`10^{-5} \geq a \geq 1.5` or :math:`10^{-5} \geq a \geq a_{max}+0.5`

            mu_vals    (array): :math:`\mu(a)` values on a_grid

        Returns:
            array: D(a), f(a)
        """
        # Pass interpolation arrays directly to MG_D_derivatives
        D, dDda = odeint(
            MG_D_derivatives,
            jnp.array([self.a_start, 1.0]),
            self.aa,
            a_grid,
            mu_vals,
            self.omega0,
            self.w0,
            self.wa,
        ).T
        f = self.aa * dDda / D
        return D[1:], f[1:]


class LCDM(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def growth_parameters(self, rtol=1.4e-8, atol=1.4e-8):
        """
        Computes growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation in the :math:`\Lambda` CDM cosmology.

        Args:
            rtol: Relative tolerance for ODE solver (default 1.4e-8). Use larger values (e.g., 1e-6) for faster computation.
            atol: Absolute tolerance for ODE solver (default 1.4e-8). Use larger values (e.g., 1e-6) for faster computation.

        Returns:
            array: D(a), f(a)
        """
        # Use optimized LCDM-specific function
        D, dDda = odeint(
            D_derivatives_LCDM,
            jnp.array([self.a_start, 1.0]),
            self.aa,
            self.omega0,
            rtol=rtol,
            atol=atol,
        ).T
        f = self.aa * dDda / D
        return D[1:], f[1:]


class wCDM(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def growth_parameters(self, rtol=1.4e-8, atol=1.4e-8):
        """
        Computes growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation in the wCDM cosmology with :math:`w=w_0` specialised by the initalisation.

        Args:
            rtol: Relative tolerance for ODE solver (default 1.4e-8). Use larger values (e.g., 1e-6) for faster computation.
            atol: Absolute tolerance for ODE solver (default 1.4e-8). Use larger values (e.g., 1e-6) for faster computation.

        Returns:
            array: D(a), f(a)
        """

        D, dDda = odeint(
            D_derivatives,
            jnp.array([self.a_start, 1.0]),
            self.aa,
            self.omega0,
            self.w0,
            0.0,
            rtol=rtol,
            atol=atol,
        ).T
        f = self.aa * dDda / D
        return D[1:], f[1:]


class w0waCDM(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def growth_parameters(self, rtol=1.4e-8, atol=1.4e-8):
        """
        Computes growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation in the :math:`w_0 w_a` CDM cosmology.

        Args:
            rtol: Relative tolerance for ODE solver (default 1.4e-8). Use larger values (e.g., 1e-6) for faster computation.
            atol: Absolute tolerance for ODE solver (default 1.4e-8). Use larger values (e.g., 1e-6) for faster computation.

        Returns:
            array: D(a), f(a)
        """

        D, dDda = odeint(
            D_derivatives,
            jnp.array([self.a_start, 1.0]),
            self.aa,
            self.omega0,
            self.w0,
            self.wa,
            rtol=rtol,
            atol=atol,
        ).T
        f = self.aa * dDda / D
        return D[1:], f[1:]


class IDE(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def growth_parameters(self, xi=0.0):
        """
        Computes growth factor :math:`D` and growth rate :math:`f`
        at the scale factor specified by initialisation in the :math:`w_0 w_a A` CDM cosmology, also
        known as interacting dark energy. In this abbreviation :math:`w_0 w_a` are the parameters from the
        equation of state for dark energy: :math:`w(a) = w_0 + w_a (1-a)` and parameter :math:`A = (1+w(a)) \\xi`
        is introduced in such way to allow us to sample the parameters space with clear definition of the
        :math:`w(a) \sim -1` case.

        Args:
        xi       (scalar): scattering strength

        Returns:
            array: D(a), f(a)
        """

        assert xi >= 0
        D, dDda = odeint(
            IDE_D_derivatives,
            jnp.array([self.a_start, 1.0]),
            self.aa,
            self.omega0,
            self.h,
            self.w0,
            self.wa,
            xi,
        ).T
        f = self.aa * dDda / D
        return D[1:], f[1:]


class fR_HS(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def growth_parameters(self, k_arr, fR0=1e-9):
        """
        Computes scale-dependent growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation in a f(R) gravity in the :math:`n=1` Hu-Sawicki model
        :math:`f(R) = -m^2 \\frac{c_1 (R/m^2)^n}{c_2(R/m^2)+1}` with :math:`\mu(a, k) = 1 + \\frac{1}{3} \\frac{k^2}{k^2 + \\frac{a^2}{3 k^2}}`
        and :math:`f_{RR}(a) = \\frac{n(n+1)}{m^2} \\frac{c_1}{c_2^2} \left( \\frac{m^2}{R}  \\right)^{n+2}`
        where :math:`\\frac{c_1}{c_2^2} = -\\frac{1}{n} \left( 1 + \\frac{4 \Omega_\mathrm{DE,0}}{\Omega_\mathrm{m,0}} \\right)^{n+1} f_{R0}`,
        :math:`R(a) = m^2 \left( \\frac{3}{a^3} + 2 \\frac{c_1}{c_2} \\right)`, :math:`\\frac{c_1}{c_2} = 6 \\frac{\Omega_\mathrm{DE,0}}{\Omega_\mathrm{m,0}}`
        and :math:`m^2 = H_0^2 \Omega_\mathrm{m,0}`.
        For :math:`\Lambda` CDM scenario :math:`f_{R0} = 0`, typical scenarios include a weak deviation
        with :math:`f_{R0} = -10^{-6}` and a stronger deviation with :math:`f_{R0} = -10^{-5}`.
        Only :math:`\Lambda` CDM expansion is allowed due to the model specifics.

        Args:
            fR0       (positive float):  absolute value of the modification at :math:`a=1`, a positive number between :math:`10^{-9}` and :math:`10^{-2}`

            k         (array):   scales in :math:`h/` Mpc

        Returns:
            array: D(k, a), f(k, a)
        """

        assert fR0 > 0.0
        c0 = 1.0 / 2997.92458  # H = h/Mpc c0
        var1 = [(k_i / self.aa_interp) ** 2 for k_i in k_arr]
        mu_fR = jnp.array(
            [
                1.0
                + 1.0
                / 3
                * var1_i
                / (
                    var1_i
                    + (self.omega0 / self.aa_interp**3 - 4.0 * (self.omega0 - 1.0)) ** 3
                    / (2.0 * (4.0 - 3.0 * self.omega0) ** 2 * fR0 / c0**2)
                )
                for var1_i in var1
            ]
        )

        # Pass interpolation arrays directly to MG_D_derivatives for each k
        D_f_i = [
            odeint(
                MG_D_derivatives,
                jnp.array([self.a_start, 1.0]),
                self.aa,
                self.aa_interp,
                mu_fR[i, :],
                self.omega0,
                -1.0,
                0.0,
            ).T
            for i in range(len(k_arr))
        ]
        D = jnp.array([D_i for D_i, _ in D_f_i])
        f = jnp.array([self.aa * dDda_i / D_i for D_i, dDda_i in D_f_i])

        return D[:, 1:], f[:, 1:]


class nDGP(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def beta(self, a, omegarc):
        """
        Computes :math:`\\beta(a) = 1 + \\frac{E(a)}{\sqrt{\Omega_\mathrm{rc}}} \left( 1 + \\frac{1}{3}\\frac{\mathrm{d} \ln{H}}{\mathrm{d} \ln{a}}  \\right)`
        where :math:`\Omega_\mathrm{rc} = \\frac{1}{4 H_0^2 r_c^2}`.

        Args:
            a         (array):   values of scalar factors

            omegarc   (float):  value of the modification, a positive number between 1.e-6 and 1.e6

        Returns:
            array: beta(a)
        """

        omegaL = (
            (1.0 - self.omega0)
            * a ** (-3.0 * (1.0 + self.w0 + self.wa))
            * jnp.exp(3.0 * (-1.0 + a) * self.wa)
        )
        E = jnp.sqrt(self.omega0 / a**3 + omegaL)
        return 1.0 + E / jnp.sqrt(omegarc) * (
            1.0 + 1.0 / 3 * dlnH_dlna(a, self.omega0, self.w0, self.wa)
        )

    def growth_parameters(self, omegarc=1.0e-6):
        """
        Computes growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation for a nDGP cosmology with :math:`\mu(a) = 1 + \\frac{1}{3\\beta(a)}`.
        For :math:`\Lambda` CDM scenario :math:`\Omega_\mathrm{rc} = 0`, typical scenarios include a weak deviation
        with :math:`\Omega_\mathrm{rc} = 0.01` for :math:`r_c = 5 H_0^{-1}` and a stronger deviation with
        :math:`\Omega_\mathrm{rc} = 0.25` for :math:`r_c = H_0^{-1}`.

        Args:
            omegarc   (float):  value of the modification, a positive number between 1.e-6 and 1.e6

        Returns:
            array: D(a), f(a)
        """

        mu_nDGP = 1.0 + 1.0 / (3.0 * self.beta(self.aa_interp, omegarc))

        # Pass interpolation arrays directly to MG_D_derivatives
        D, dDda = odeint(
            MG_D_derivatives,
            jnp.array([self.a_start, 1.0]),
            self.aa,
            self.aa_interp,
            mu_nDGP,
            self.omega0,
            self.w0,
            self.wa,
        ).T
        f = self.aa * dDda / D
        return D[1:], f[1:]


class Linder_gamma(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def growth_parameters(self, gamma=0.55, use_quad=True):
        """
        Computes growth rate :math:`f = \Omega_\mathrm{m}(a)^\gamma` and
        :math:`D(a)=D_\mathrm{ini} \exp{\int_{a_\mathrm{ini}}^a \mathrm{d}\\tilde{a} f(\\tilde{a})/\\tilde{a} }`
        with :math:`D_\mathrm{ini} = D_{\Lambda \mathrm{CDM}} (a=10^{-4})`
        at the scale factor specified by initialisation.

        Args:
            gamma (float): growth index, equals 0.55 in standard cosmology
            use_quad (bool): If True, use scipy.quad for maximum accuracy (slower).
                           If False, use Simpson's rule with n_points (faster, still accurate).

        Returns:
            array: D(a), f(a)
        """

        if use_quad:
            f = Omega_m(self.aa[1:], self.omega0, self.w0, self.wa) ** gamma

            def func(a_):
                return Omega_m(a_, self.omega0, self.w0, self.wa) ** gamma / a_

            Dini = self.a_start * hyp2f1(
                1.0 / 3,
                1.0,
                11.0 / 6,
                (self.omega0 - 1.0) / self.omega0 * self.a_start**3,
            )
            # Use quad for maximum accuracy (as in original)
            D = jnp.array(
                [
                    (Dini * jnp.exp(quad(func, self.a_start, a_i)[0]))
                    for a_i in self.aa[1:]
                ]
            )
            return D, f
        else:
            Omm = Omega_m(self.aa_interp, self.omega0, self.w0, self.wa)
            # mu_gamma = 2./3.*Omm**(gamma-1.)*(Omm**gamma+2.-3.*gamma+3.*(gamma-0.5)*Omm) for LCDM background
            A = -3.0 * (1.0 + self.w0 + self.wa)
            omegaf = self.aa_interp**A * jnp.exp(
                3.0 * (-1.0 + self.aa_interp) * self.wa
            )
            omegaL = (1.0 - self.omega0) * omegaf
            E2 = self.omega0 / self.aa_interp**3 + omegaL
            var1 = (
                -3.0 * Omm
                + (3.0 * self.aa_interp * self.wa - 3.0 * (1.0 + self.w0 + self.wa))
                * omegaL
                / E2
            )  # a/E^2 dE^2/da
            mu_gamma = (
                2.0
                / 3.0
                * Omm ** (gamma - 1.0)
                * (Omm**gamma + 2.0 - 3.0 * gamma + (0.5 - gamma) * var1)
            )
            D, dDda = odeint(
                MG_D_derivatives,
                jnp.array([self.a_start, 1.0]),
                self.aa,
                self.aa_interp,
                mu_gamma,
                self.omega0,
                self.w0,
                self.wa,
            ).T
            f = self.aa * dDda / D
            return D[1:], f[1:]


class Linder_gamma_a(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def growth_parameters(self, gamma0=0.55, gamma1=0.0, use_quad=True):
        """
        Computes growth rate :math:`f = \Omega_\mathrm{m}(a)^{\gamma(a)}` and
        :math:`D(a)=D_\mathrm{ini} \exp{\int_{a_\mathrm{ini}}^a \mathrm{d}\\tilde{a} f(\\tilde{a})/\\tilde{a} }`
        with :math:`D_\mathrm{ini} = D_{\Lambda \mathrm{CDM}} (a=10^{-4})`
        at the scale factor specified by initialisation. The time parameterisation of
        the growth index is taken from arXiv: 2304.07281,
        namely :math:`\gamma(a) = \gamma_0 + \gamma_1 \\frac{(1-a)^2}{a}`.

        Args:
            gamma0 (float): growth index, equals 0.55 in standard cosmology

            gamma1 (float): growth index time component, equals 0 in standard cosmology

            use_quad (bool): If True, use scipy.quad for maximum accuracy (slower).
                           If False, use Simpson's rule with n_points (faster, still accurate).

        Returns:
            array: D(a), f(a)
        """
        if use_quad:
            f = Omega_m(self.aa[1:], self.omega0, self.w0, self.wa) ** (
                gamma0 + gamma1 * (1.0 - self.aa[1:]) ** 2 / self.aa[1:]
            )

            def func(a_):
                return (
                    Omega_m(a_, self.omega0, self.w0, self.wa)
                    ** (gamma0 + gamma1 * (1.0 - a_) ** 2 / a_)
                    / a_
                )

            Dini = self.a_start * hyp2f1(
                1.0 / 3,
                1.0,
                11.0 / 6,
                (self.omega0 - 1.0) / self.omega0 * self.a_start**3,
            )
            # Use quad for maximum accuracy (as in original)
            D = jnp.array(
                [
                    Dini * jnp.exp(quad(func, self.a_start, a_i)[0])
                    for a_i in self.aa[1:]
                ]
            )
            return D, f
        else:
            Omm = Omega_m(self.aa_interp, self.omega0, self.w0, self.wa)
            gamma = gamma0 + gamma1 * (self.aa_interp + 1.0 / self.aa_interp - 2.0)
            # mu_gamma = 2./3.*Omm**(gamma-1.)*(Omm**gamma+2.-3.*gamma+3.*(gamma-0.5)*Omm) for LCDM background
            A = -3.0 * (1.0 + self.w0 + self.wa)
            omegaf = self.aa_interp**A * jnp.exp(
                3.0 * (-1.0 + self.aa_interp) * self.wa
            )
            omegaL = (1.0 - self.omega0) * omegaf
            E2 = self.omega0 / self.aa_interp**3 + omegaL
            var1 = (
                -3.0 * Omm
                + (3.0 * self.aa_interp * self.wa - 3.0 * (1.0 + self.w0 + self.wa))
                * omegaL
                / E2
            )  # a/E^2 dE^2/da
            mu_gamma = (
                2.0
                / 3.0
                * Omm ** (gamma - 1.0)
                * (Omm**gamma + 2.0 - 3.0 * gamma + (0.5 - gamma) * var1)
            )
            mu_gamma_a = (
                2.0
                / 3.0
                * Omm ** (gamma - 1.0)
                * gamma1
                * (self.aa_interp - 1.0 / self.aa_interp)
                * jnp.log(Omm)
                + mu_gamma
            )
            D, dDda = odeint(
                MG_D_derivatives,
                jnp.array([self.a_start, 1.0]),
                self.aa,
                self.aa_interp,
                mu_gamma_a,
                self.omega0,
                self.w0,
                self.wa,
            ).T
            f = self.aa * dDda / D
            return D[1:], f[1:]

# cloelite imports
from numpy import ndarray
from cloelite.cosmology.cosmology import Background
from cloelite.cosmology.cosmology import Perturbations

# General imports
import jax.numpy as np
import jax
import jax.lax as lx
import functools

"""
## Author:
    **Name**: M. Bonici & G. Canas-Herrea
    **Date**: June 9, 2024

## Notes:

- Make it completely differentiable

"""

class JAXBackground(Background):
    def __init__(self, H0: float, Omb: float, Omc: float, Omk: float, As: float, ns: float,
                 w: float, wa: float, gamma_MG: float):
        r"""
        A class to define background cosmology using JAX
        and inheriting from Cosmology parent class

        """
        self.H0 = float(H0)
        self.Omb = float(Omb)
        self.Omc = float(Omc)
        self.Omk = float(Omk)
        self.As = float(As)
        self.ns = float(ns)
        self.w = float(w)
        self.wa = float(wa)
        self.gamma_MG = float(gamma_MG)

    def hubble_parameter(self, zs) -> np.ndarray:
        r"""
        Retrieves the hubble parameter as
        a function of redshift

        .. math::
            H(z) = \sqrt

        Parameters
        ----------
        zs: numpy.ndarray
            Redshifts for the matter density

        Returns
        -------
        Hubble parameter: numpy.ndarray
            hubble parameter as a function of redshift

        """
        return self.H0 * np.sqrt((self.Omb+self.Omc)*np.power(1+zs, 3) +
                                 (self.Omk)*np.power(1+zs, 2) +
                                 (1-self.Omb-self.Omc-self.Omk) * np.power(1+zs, 3*(1+self.w+self.wa))*np.exp(-3*self.wa*zs/(1+zs)))

    #@property
    def comoving_distance(self, zs) -> np.ndarray:
        """
        Calculates the comoving distance for given redshifts.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the comoving distance.

        Returns:
        --------
        np.ndarray
            The comoving distance as a function of redshift.
        """
        c_0 = 2.99792458e5 #please, put all the constanst in a single place
        fun = lambda x: 1/self.hubble_parameter(x)
        y = simps(fun, 0, zs, N=512) * c_0
        return y

    def matter_density(self, zs) -> np.ndarray:
        r"""
        Computes the matter density as

        .. math::
            \Omega_{\rm m}(z) = \Omega_{{\rm m},0}(1+z)^3H_0^2/H^2(z)

        Parameters
        ----------
        zs: numpy.ndarray
            Redshifts at which to calculate the matter density

        Returns
        -------
        Matter density parameter: numpy.ndarray
            Matter density as a function of redshift

        """

        return

    #@property
    def transverse_comoving_distance(self, zs) -> np.ndarray:
        """
        Calculates the transverse comoving distance for given redshifts.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the transverse comoving distance.

        Returns:
        --------
        np.ndarray
            The transverse comoving distance as a function of redshift.
        """
        return self.comoving_distance(zs)
    #TODO add the lax conditionals to account for the curvature!

    #@property
    def angular_diameter_distance(self, zs) -> np.ndarray:
        """
        Calculates the angular diameter distance for given redshifts.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the angular diameter distance.

        Returns:
        --------
        np.ndarray
            The angular diameter distance as a function of redshift.
        """
        return self.transverse_comoving_distance(zs)/(1+zs)

class JAXPerturbations(Perturbations):
    def __init__(self, H0: float, Omb: float, Omc: float, Omk: float, As: float, ns: float,
                 w: float, wa: float, gamma_MG: float):
        r"""
        A class to define perturbations cosmology using JAX
        and inheriting from Cosmology parent class

        """
        self.H0 = float(H0)
        self.Omb = float(Omb)
        self.Omc = float(Omc)
        self.Omk = float(Omk)
        self.As = float(As)
        self.ns = float(ns)
        self.w = float(w)
        self.wa = float(wa)
        self.gamma_MG = float(gamma_MG)

    def w_a(self, a):
        return self.w + (1.0 - a) * self.wa  # Equation (6) in Linder (2003)



    def f_de(self, a):
        return -3.0 * (1.0 + self.w + self.wa) * np.log(a) + 3.0 * self.wa * (a - 1.0)

    def Esqr(self, a):
        Omm = self.Omb + self.Omc
        OmDE = 1. - Omm - self.Omk
        return (Omm * np.power(a, -3) + self.Omk * np.power(a, -2)
                + OmDE * np.exp(self.f_de(a))
    )

    def Omega_m_a(self, a):
        Omm = self.Omb + self.Omc
        return Omm * np.power(a, -3) / self.Esqr(a)

    def Omega_de_a(self, a):
        OmDE = 1. - self.Omb - self.Omc - self.Omk
        return OmDE * np.exp(self.f_de(a)) / self.Esqr(a)

    def D_derivs(self, y, x):
            q = (2.0 - 0.5* ( self.Omega_m_a(x) + (1.0 + 3.0 * self.w_a(x)) *
                             self.Omega_de_a(x))) / x
            r = 1.5 * self.Omega_m_a(x) / x / x
            return np.array([y[1], -q * y[1] + r * y[0]])

    def growth_factor(self, zs):
        atab = np.logspace(-3., 0.0, 128)

        a_s = a_z(zs)

        y0 = np.array([atab[0], 1.0])
        fn = lambda x, y : self.D_derivs(x,y)
        y = odeint(fn, y0, atab)
        y1 = y[:, 0]
        gtab = y1 / y1[-1]

        result = interp(a_s, atab, gtab)

        return result

    def growth_rate(self, zs, ks) -> np.ndarray:
        return

    def linear_matter_power_spectrum(self):
        return

#function takesnfrom JAXCosmo. Should likely be moved to an utils.py
def simps(f, a, b, N=128):
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S

#function takes from JAXCosmo. Should likely be moved to an utils.py
def odeint(fn, y0, t):

    def rk4(carry, t):
        y, t_prev = carry
        h = t - t_prev
        k1 = fn(y, t_prev)
        k2 = fn(y + h * k1 / 2, t_prev + h / 2)
        k3 = fn(y + h * k2 / 2, t_prev + h / 2)
        k4 = fn(y + h * k3, t)
        y = y + 1.0 / 6.0 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t), y

    (yf, _), y = lx.scan(rk4, (y0, np.array(t[0])), t)
    return y

@functools.partial(jax.vmap, in_axes=(0, None, None))
def interp(x, xp, fp):
    """
    Simple equivalent of np.interp that compute a linear interpolation.

    We are not doing any checks, so make sure your query points are lying
    inside the array.

    TODO: Implement proper interpolation!

    x, xp, fp need to be 1d arrays
    """
    # First we find the nearest neighbour
    ind = np.argmin((x - xp) ** 2)

    # Perform linear interpolation
    ind = np.clip(ind, 1, len(xp) - 2)

    xi = xp[ind]
    # Figure out if we are on the right or the left of nearest
    s = np.sign(np.clip(x, xp[1], xp[-2]) - xi).astype(np.int32)
    a = (fp[ind + np.copysign(1, s).astype(np.int32)] - fp[ind]) / (
        xp[ind + np.copysign(1, s).astype(np.int32)] - xp[ind]
    )
    b = fp[ind] - a * xp[ind]
    return a * x + b

@jax.jit
def a_z(z):
    return 1/(1+z)

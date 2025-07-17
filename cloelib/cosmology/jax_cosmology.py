"""
Implementation of Background and Perturbation cosmology using JAX.

The module enables automatic differentiation.
All of the functions are completely differentiable.
"""

# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Background
from cloelib.cosmology.cosmology import Perturbations
#from cloelib.cosmology.cosmology import NonLinearPerturbations

# General imports
import jax.numpy as np
import jax
import jax.lax as lx
import functools
import interpax
from quadax import quadgk

class JAXBackground:
    """Class to define background cosmology using JAX,inheriting from Cosmology parent class."""

    def __init__(self, H0: float, Omega_b0: float, Omega_cdm0: float, Omega_k0: float,
                 As: float, ns: float, mnu: float,
                 w0: float, wa: float, gamma_MG: float):
        """
        Initialize the JAXBackground class.

        Args:
            H0 (float): Hubble parameter in [km/s/Mpc].
            Omega_b0 (float): Baryonic matter density parameter.
            Omega_cdm0 (float): Cold dark matter density parameter.
            Omega_k0(float): Curvature density parameter.
            As (float): Scalar amplitude of primordial fluctuations.
            ns (float): Scalar spectral index.
            mnu (float): Total sum of neutrino mass in [eV].
            w0 (float): Equation of state parameter for dark energy.
            wa (float): Time evolution of the dark energy equation of state.
            gamma_MG (float): Modified gravity growth parameter.

        """
        self.H0 = H0
        self.h = self.H0 / 100
        self.Omega_b0 = Omega_b0
        self.Omega_cdm0 = Omega_cdm0
        self.Omega_k0 = Omega_k0
        self.As = As
        self.ns = ns
        self.w0 = w0
        self.wa = wa
        self.gamma_MG = gamma_MG
        self.mnu = mnu
        self.Omega_nu0 = self.mnu/(93.14*(self.h)**2)#this is a semplification, we are assuming
        #neutrinos are non relativistic
        self.Omega_m0 = self.Omega_b0+self.Omega_cdm0+self.Omega_nu0
        self.sigma_8 = As_to_sigma8_max_precision(self.As, self.Omega_m0,
                                                  self.Omega_b0, self.h, self.ns, 0.,
                                                  self.w0, self.wa)

        # Initialize JaxBgk parameters
        self.interface_args = {'JAXparams': {}}  # Use a dictionary for CLASS parameters
        self.interface_args['JAXparams']['H0'] = self.H0
        self.interface_args['JAXparams']['Omega_b'] = self.Omega_b0
        self.interface_args['JAXparams']['Omega_cdm'] = self.Omega_cdm0
        self.interface_args['JAXparams']['Omega_k'] = self.Omega_k0
        self.interface_args['JAXparams']['Omega_m0'] = self.Omega_m0
        self.interface_args['JAXparams']['Omega_nu0'] = self.Omega_nu0
        self.interface_args['JAXparams']['n_s'] = self.ns
        self.interface_args['JAXparams']['m_ncdm'] = self.mnu
        self.interface_args['JAXparams']['A_s'] = self.As
        self.interface_args['JAXparams']['w0_fld'] = self.w0 # or w0
        self.interface_args['JAXparams']['wa_fld'] = self.wa # or wa
        self.interface_args['JAXparams']['sigma_8'] = self.sigma_8 # or wa

    @property
    def _interface_args(self) -> dict:
        """Save internal structure format of interface codes."""
        return self.interface_args

    def hubble_parameter(self, zs, units: str = "km/s/Mpc") -> np.ndarray:
        """
        Return the Hubble parameter as a function of redshift.

        Args:
            zs : Redshifts.
            units (str): Units for the Hubble parameter ('1/Mpc' or 'km/s/Mpc').

        Returns:
            Hubble parameter values at specified redshift(s).

        """
        c_0 = SPEED_OF_LIGHT / 1000
        Omega_m0 = self.Omega_b0+self.Omega_cdm0+self.mnu/(93.14*(self.H0/100)**2)
        x = self.H0 * np.sqrt(Omega_m0*np.power(1+zs, 3) +
                                 (self.Omega_k0)*np.power(1+zs, 2) +
                                 (1-Omega_m0-self.Omega_k0) * np.power(1+zs, 3*(1+self.w0+self.wa))*np.exp(-3*self.wa*zs/(1+zs)))

        def default_case(x):
            return x

        def one_Mpc_case(x):
            return x/c_0

        conditions = np.array([units=="km/s/Mpc", units=="1/Mpc"])
        index = np.argwhere(conditions, size=1).squeeze()

        return lx.switch(index, [default_case, one_Mpc_case],x)

    def comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Calculate the comoving distance for given redshifts.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the comoving distance.

        Returns:
        --------
        np.ndarray
            The comoving distance as a function of redshift.
        """
        c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s
        fun = lambda x: 1/self.hubble_parameter(x)

        def myquad(x, fun):
            y, _ = quadgk(fun, [0., x])
            return y

        y = np.array([myquad(myz, fun)  for myz in zs])
        return y*c_0

    def transverse_comoving_distance(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the transverse comoving distance between two redshifts.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Transverse comoving distance values.
        """
        delta_z = self.comoving_distance(zs)
        p = np.concatenate([np.array([self.Omega_k0]), delta_z], axis=0)

        def default_case(p):
            return p[1:]

        def positive_case(p):
            return np.sinh(np.sqrt(p[0]) * p[1:]) / np.sqrt(p[0])

        def negative_case(p):
            return np.sin(np.sqrt(-p[0]) * p[1:]) / np.sqrt(-p[0])

        conditions = np.array([self.Omega_k0>0., self.Omega_k0<0., self.Omega_k0==0.])
        index = np.argwhere(conditions, size=1).squeeze()

        return lx.switch(index, [positive_case, negative_case, default_case],p)

    def angular_diameter_distance(self, zs) -> np.ndarray:
        """
        Calculate the angular diameter distance for given redshifts.

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

    def Omega_b(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the baryon density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Matter density values.
        """
        return np.array([self.Omega_b0 * (1+z)**3 /(self.hubble_parameter(z)/self.H0)**2  for z in zs])

    def Omega_m(self, zs: np.ndarray) -> np.ndarray:
        """
        Return the matter density as a function of redshift.

        Args:
            zs (np.ndarray): Array of redshifts.

        Returns:
            np.ndarray: Matter density values.
        """
        return np.array([(self.Omega_m0) * (1+z)**3 /(self.hubble_parameter(z)/self.H0)**2  for z in zs])

    def w_a(self, a):
        """Write documentation (TODO)."""
        return self.w0 + (1.0 - a) * self.wa  # Equation (6) in Linder (2003)

    def f_de(self, a):
        """Write documentation (TODO)."""
        return -3.0 * (1.0 + self.w0 + self.wa) * np.log(a) + 3.0 * self.wa * (a - 1.0)

    def Esqr(self, a):
        """Write documentation (TODO)."""
        OmDE = 1. - self.Omega_m0 - self.Omega_k0
        return (self.Omega_m0 * np.power(a, -3) + self.Omega_k0 * np.power(a, -2)
                + OmDE * np.exp(self.f_de(a)))

    def Omega_m_a(self, a):
        """Write documentation (TODO)."""
        return self.Omega_m0 * np.power(a, -3) / self.Esqr(a)

    def Omega_de_a(self, a):
        """Write documentation (TODO)."""
        OmDE = 1. - self.Omega_m0 - self.Omega_k0
        return OmDE * np.exp(self.f_de(a)) / self.Esqr(a)

    @property
    def rdrag(self) -> float:
        """
        Sound horizon radius at last scattering.
        """
        raise NotImplementedError("rdrag not implemented for jax yet.")


class JAXLinearPerturbations:
    """A wrapper for JAX linear perturbation calculations."""

    def __init__(self, background: Background, redshifts: np.ndarray) -> None:
        """
        Initialize the JAXLinearPerturbations class with a background instance.

        Args:
            background (Background): A Background instance.
            redshifts (np.ndarray): Array of redshifts for the calculations.
        """
        self.background = background
        self.z = redshifts

    def D_derivs(self, y, x):
        """Write documentation (TODO)."""
        q = (2.0 - 0.5* (self.background.Omega_m_a(x) + (1.0 + 3.0 * self.background.w_a(x)) *
                         self.background.Omega_de_a(x))) / x
        r = 1.5 * self.background.Omega_m_a(x) / x / x
        return np.array([y[1], -q * y[1] + r * y[0]])

    def growth_factor(self, zs):
        """Compute the growth factor."""
        atab = np.logspace(-3., 0.0, 128)

        a_s = a_z(zs)

        y0 = np.array([atab[0], 1.0])
        fn = lambda x, y : self.D_derivs(x,y)
        y = odeint(fn, y0, atab)
        y1 = y[:, 0]
        gtab = y1 / y1[-1]

        result = interp(a_s, atab, gtab)

        return result

    def growth_rate(self, zs):
        """Compute the growth rate."""
        atab = np.logspace(-3., 0.0, 256)

        a_s = a_z(zs)

        y0 = np.array([atab[0], 1.0])
        fn = lambda x, y : self.D_derivs(x,y)
        y = odeint(fn, y0, atab)
        y1 = y[:, 0]
        gtab = y1 / y1[-1]

        ftab = y[:, 1] / y1[-1] * atab / gtab

        result = interp(a_s, atab, ftab)
        return result

    def transfer_Eisenstein_Hu(self, ks):
        """Compute the Eisenstein & Hu matter transfer function.

        Parameters
        ----------
        cosmo: Background
        Background cosmology

        k: array_like
        Wave number in h Mpc^{-1}

        type: str, optional
        Type of transfer function. Either 'eisenhu' or 'eisenhu_osc'
        (def: 'eisenhu_osc')

        Returns
        -------
        T: array_like
        Value of the transfer function at the requested wave number

        Notes
        -----
        The Eisenstein & Hu transfer functions are computed using the fitting
        formulae of :cite:`1998:EisensteinHu`

        """
        #############################################
        # Quantities computed from 1998:EisensteinHu
        # Provides : - k_eq   : scale of the particle horizon at equality epoch
        #            - z_eq   : redshift of equality epoch
        #            - R_eq   : ratio of the baryon to photon momentum density
        #                       at z_eq
        #            - z_d    : redshift of drag epoch
        #            - R_d    : ratio of the baryon to photon momentum density
        #                       at z_d
        #            - sh_d   : sound horizon at drag epoch
        #            - k_silk : Silk damping scale
        T_2_7_sqr = (2.726 / 2.7) ** 2
        h2 = (self.background.H0/100) ** 2

        w_m = (self.background.Omega_m0) * h2
        w_b = self.background.Omega_b0 * h2
        fb = self.background.Omega_b0 / (self.background.Omega_m0)
        fc = (self.background.Omega_cdm0+self.background.Omega_nu0) / (self.background.Omega_m0)

        k_eq = 7.46e-2 * w_m / T_2_7_sqr / (self.background.h)  # Eq. (3) [h/Mpc]
        z_eq = 2.50e4 * w_m / (T_2_7_sqr) ** 2  # Eq. (2)

        # z drag from Eq. (4)
        b1 = 0.313 * np.power(w_m, -0.419) * (1.0 + 0.607 * np.power(w_m, 0.674))
        b2 = 0.238 * np.power(w_m, 0.223)
        z_d = (
            1291.0
            * np.power(w_m, 0.251)
            / (1.0 + 0.659 * np.power(w_m, 0.828))
            * (1.0 + b1 * np.power(w_b, b2))
        )

        # Ratio of the baryon to photon momentum density at z_d  Eq. (5)
        R_d = 31.5 * w_b / (T_2_7_sqr) ** 2 * (1.0e3 / z_d)
        # Ratio of the baryon to photon momentum density at z_eq Eq. (5)
        R_eq = 31.5 * w_b / (T_2_7_sqr) ** 2 * (1.0e3 / z_eq)
        # Sound horizon at drag epoch in h^-1 Mpc Eq. (6)
        sh_d = (
            2.0
            / (3.0 * k_eq)
            * np.sqrt(6.0 / R_eq)
            * np.log((np.sqrt(1.0 + R_d) + np.sqrt(R_eq + R_d)) / (1.0 + np.sqrt(R_eq)))
        )
        # Eq. (7) but in [hMpc^{-1}]
        k_silk = (
            1.6
            * np.power(w_b, 0.52)
            * np.power(w_m, 0.73)
            * (1.0 + np.power(10.4 * w_m, -0.95))
            / (self.background.h)
        )
        #############################################

        alpha_gamma = (
            1.0
            - 0.328 * np.log(431.0 * w_m) * w_b / w_m
            + 0.38 * np.log(22.3 * w_m) * (self.background.Omega_b0/ (self.background.Omega_cdm0 + self.background.Omega_b0 + self.background.mnu/(93.14*(self.background.H0/100)**2))) ** 2
        )
        gamma_eff = ((self.background.Omega_m0) * (self.background.h)
            * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * ks * sh_d) ** 4))
        )


        a1 = np.power(46.9 * w_m, 0.670) * (1.0 + np.power(32.1 * w_m, -0.532))
        a2 = np.power(12.0 * w_m, 0.424) * (1.0 + np.power(45.0 * w_m, -0.582))
        alpha_c = np.power(a1, -fb) * np.power(a2, -(fb**3))
        b1 = 0.944 / (1.0 + np.power(458.0 * w_m, -0.708))
        b2 = np.power(0.395 * w_m, -0.0266)
        beta_c = 1.0 + b1 * (np.power(fc, b2) - 1.0)
        beta_c = 1.0 / beta_c

        # EH98 (19). [k] = h/Mpc
        def T_tilde(k1, alpha, beta):
            # EH98 (10); [q] = 1 BUT [k] = h/Mpc
            q = k1 / (13.41 * k_eq)
            L = np.log(np.exp(1.0) + 1.8 * beta * q)
            C = 14.2 / alpha + 386.0 / (1.0 + 69.9 * np.power(q, 1.08))
            T0 = L / (L + C * q * q)
            return T0

        # EH98 (17, 18)
        f = 1.0 / (1.0 + (ks * sh_d / 5.4) ** 4)
        Tc = f * T_tilde(ks, 1.0, beta_c) + (1.0 - f) * T_tilde(ks, alpha_c, beta_c)

        # Baryon transfer function
        # EH98 (19, 14, 21)
        y = (1.0 + z_eq) / (1.0 + z_d)
        x = np.sqrt(1.0 + y)
        G_EH98 = y * (-6.0 * x + (2.0 + 3.0 * y) * np.log((x + 1.0) / (x - 1.0)))
        alpha_b = 2.07 * k_eq * sh_d * np.power(1.0 + R_d, -0.75) * G_EH98

        beta_node = 8.41 * np.power(w_m, 0.435)
        tilde_s = sh_d / np.power(1.0 + (beta_node / (ks * sh_d)) ** 3, 1.0 / 3.0)

        beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * np.sqrt((17.2 * w_m) ** 2 + 1.0)

        # [tilde_s] = Mpc/h
        Tb = (
            T_tilde(ks, 1.0, 1.0) / (1.0 + (ks * sh_d / 5.2) ** 2)
            + alpha_b
            / (1.0 + (beta_b / (ks * sh_d)) ** 3)
            * np.exp(-np.power(ks / k_silk, 1.4))
        ) * np.sinc(ks * tilde_s / np.pi)

        # Total transfer function
        res = fb * Tb + fc * Tc

        return res

    def primordial_matter_power(self, ks):
        """Primordial power spectrum.

        Pk = k^n
        """
        return ks ** self.background.ns

    def sigmasqr(self, R, kmin=0.0001, kmax=1000.0, ksteps=5):
        r"""Compute the energy of the fluctuations within a sphere of R h^{-1} Mpc.

        .. math::

        \\sigma^2(R)= \\frac{1}{2 \\pi^2} \\int_0^\\infty \\frac{dk}{k} k^3 P(k,z) W^2(kR)

        where

        .. math::

        W(kR) = \\frac{3j_1(kR)}{kR}
        """

        def int_sigma(logk):
            k = np.exp(logk)
            x = k * R
            w = 3.0 * (np.sin(x) - x * np.cos(x)) / (x * x * x)
            pk = self.transfer_Eisenstein_Hu(k) ** 2 * self.primordial_matter_power(k)
            return k * (k * w) ** 2 * pk

        y = romb(int_sigma, np.log10(kmin), np.log10(kmax), divmax=7)

        return 1.0 / (2.0 * np.pi**2.0) * y

    def sigma8sqr(self, kmin=0.0001, kmax=100.0):
        r"""Compute the energy of the fluctuations within a sphere of R h^{-1} Mpc.

        .. math::

        \\sigma^2(R)= \\frac{1}{2 \\pi^2} \\int_0^\\infty \\frac{dk}{k} k^3 P(k,z) W^2(kR)

        where

        .. math::

        W(kR) = \\frac{3j_1(kR)}{kR}
        """
        R = 8

        def int_sigma(logk):
            k = np.exp(logk)
            x = k * R
            w = 3.0 * (np.sin(x) - x * np.cos(x)) / (x * x * x)
            pk = self.transfer_Eisenstein_Hu(k) ** 2 * self.primordial_matter_power(k)
            return k * (k * w) ** 2 * pk

        y = simps(int_sigma, np.log10(kmin), np.log10(kmax), N = 256)
        return 1.0 / (2.0 * np.pi**2.0) * y

    def matter_power_spectrum(self, zs, ks,  hubble_units=False, k_hunit=False):
        r"""Compute the linear matter power spectrum.

        Parameters
        ----------
        zs: array_like, optional
            Redshifts

        k: array_like
            Wave number in h Mpc^{-1}

        Returns
        -------
        pk: array_like
            Linear matter power spectrum at the specified scale
            and scale factor.

        """
        h = self.background.h

        def k_units_case(k):
            return k/h

        def kh_units_case(k):
            return k

        conditions = np.array([k_hunit==False, k_hunit==True])
        index = np.argwhere(conditions, size=1).squeeze()

        ks = np.atleast_1d(ks)
        ks = lx.switch(index, [k_units_case, kh_units_case], ks)
        zs = np.atleast_1d(zs)
        g = self.growth_factor(zs)
        t = self.transfer_Eisenstein_Hu(ks)

        pknorm = self.background.sigma_8**2 / self.sigma8sqr()#previously self.sigmasqr(8.0)
        # this means we have a 0.01% difference compared to the romberg calculation,
        # but it is much faster

        pk =  np.outer(g**2, self.primordial_matter_power(ks) * t**2)

        def hMpc_units_case(h):
            return h*h*h

        def Mpc_units_case(h):
            return 1.

        conditions = np.array([hubble_units==False, hubble_units==True])
        index = np.argwhere(conditions, size=1).squeeze()

        factor = lx.switch(index, [hMpc_units_case, Mpc_units_case], h)

        # Apply normalisation
        pk = pk * pknorm/factor
        return pk.squeeze()

class JAXNonLinearPerturbations(Perturbations):
    """Class for perturbations cosmology using JAX, inheriting from Cosmology parent class."""

    def __init__(self, linearperturbations : Perturbations):
        """Initialse the class instance."""
        self.linearperturbations = linearperturbations

    def _halofit_parameters(self, zs):
        """Compute the non linear scale, effective spectral index, spectral curvature."""
        # Step 1: Finding the non linear scale for which sigma(R)=1
        # That's our search range for the non linear scale
        logr = np.linspace(np.log(1e-4), np.log(1e1), 256)

        # TODO: implement a better root finding algorithm to compute the non linear scale
        @jax.vmap
        def R_nl(zs):
            def int_sigma(logk):
                k = np.exp(logk)
                r = np.exp(logr)
                y = np.outer(k, r)
                pk = self.linearperturbations.matter_power_spectrum(0., k)
                g = self.linearperturbations.growth_factor(np.atleast_1d(zs))
                return (
                    np.expand_dims(pk * k**3, axis=1)
                    * np.exp(-(y**2))
                    / (2.0 * np.pi**2)
                    * g**2
                )

            sigma = simps(int_sigma, np.log(1e-4), np.log(1e4), 256)
            root = interp(np.atleast_1d(1.0), sigma, logr)
            return np.exp(root).clip(
                1e-6
            )  # To ensure that the root is not too close to zero

        # Compute non linear scale
        k_nl = 1.0 / R_nl(np.atleast_1d(zs)).squeeze()


        # Step 2: Retrieve the spectral index and spectral curvature
        def integrand(logk):
            k = np.exp(logk)
            y = np.outer(k, 1.0 / k_nl)
            pk = self.linearperturbations.matter_power_spectrum(0., k)
            g = np.expand_dims(self.linearperturbations.growth_factor(np.atleast_1d(zs)), 0)
            res = (
                np.expand_dims(pk * k**3, axis=1)
                * np.exp(-(y**2))
                * g**2
                / (2.0 * np.pi**2)
            )
            dneff_dlogk = 2 * res * y**2
            dC_dlogk = 4 * res * (y**2 - y**4)
            return np.stack([dneff_dlogk, dC_dlogk], axis=1)

        res = simps(integrand, np.log(1e-4), np.log(1e4), 256)

        n_eff = res[0] - 3.0
        C = res[0] ** 2 + res[1]
        return k_nl, n_eff, C

    def halofit(self, zs, ks, hubble_units=False, k_hunit=False):
        """Write documentation (TODO)."""
        zs = np.atleast_1d(zs)
        a_s = a_z(zs)

        # Compute the linear power spectrum
        pklin = self.linearperturbations.matter_power_spectrum(zs, ks, hubble_units, k_hunit)

        # Compute non linear scale, effective spectral index and curvature
        k_nl, n, C = self._halofit_parameters(zs)

        om_m = self.linearperturbations.background.Omega_m_a(a_s)
        om_de = self.linearperturbations.background.Omega_de_a(a_s)
        w = self.linearperturbations.background.w_a(a_s)
        frac = om_de / (1.0 - om_m)

        a_n = 10 ** (
            1.5222
            + 2.8553 * n
            + 2.3706 * n**2
            + 0.9903 * n**3
            + 0.2250 * n**4
            - 0.6038 * C
            + 0.1749 * om_de * (1 + w)
        )
        b_n = 10 ** (
            -0.5642
            + 0.5864 * n
            + 0.5716 * n**2
            - 1.5474 * C
            + 0.2279 * om_de * (1 + w)
        )
        c_n = 10 ** (0.3698 + 2.0404 * n + 0.8161 * n**2 + 0.5869 * C)
        gamma_n = 0.1971 - 0.0843 * n + 0.8460 * C
        alpha_n = np.abs(6.0835 + 1.3373 * n - 0.1959 * n**2 - 5.5274 * C)
        beta_n = (
            2.0379
            - 0.7354 * n
            + 0.3157 * n**2
            + 1.2490 * n**3
            + 0.3980 * n**4
            - 0.1682 * C
        )
        mu_n = 0.0
        nu_n = 10 ** (5.2105 + 3.6902 * n)


        f1a = om_m ** (-0.0732)
        f2a = om_m ** (-0.1423)
        f3a = om_m ** (0.0725)
        f1b = om_m ** (-0.0307)
        f2b = om_m ** (-0.0585)
        f3b = om_m ** (0.0743)


        f1 = f1b
        f2 = f2b
        f3 = f3b


        f = lambda x: x / 4.0 + x**2 / 8.0

        d2l = ks**3 * pklin / (2.0 * np.pi**2)

        y = ks / k_nl

        # Eq C2
        d2q = d2l * ((1.0 + d2l) ** beta_n / (1 + alpha_n * d2l)) * np.exp(-f(y))
        d2hprime = (
            a_n * y ** (3 * f1) / (1.0 + b_n * y**f2 + (c_n * f3 * y) ** (3.0 - gamma_n))
        )
        d2h = d2hprime / (1.0 + mu_n / y + nu_n / y**2)
        # Eq. C1
        d2nl = d2q + d2h
        pk_nl = 2.0 * np.pi**2 / ks**3 * d2nl
        return pk_nl.squeeze()

    def matter_power_spectrum(self, zs, ks, hubble_units=False, k_hunit=False):
        """Compute the non-linear matter power spectrum.

        This function is just a wrapper over several nonlinear power spectra.
        """
        return jax.vmap(self.halofit, in_axes = (0, None, None, None))(zs, ks, hubble_units, k_hunit)

    def nonlinear_matter_power_spectrum_limber_grid(self, z_l, ks, zs, ells):
        """Write documentation (TODO)."""
        Pk = jax.vmap(self.nonlinear_matter_power_spectrum,
                      in_axes = (0, None))(ks, zs)
        chi = self.linearperturbations.linearperturbations.background.comoving_distance(zs)
        k_lz = np.expand_dims((ells + 0.5), 1) / chi
        Pkl = Pkl_interp_vmap(k_lz, z_l, ks, zs, Pk)
        return Pkl

#function takenfrom JAXCosmo. Should likely be moved to an utils.py
def simps(f, a, b, N=128):
    """Write documentation (TODO)."""
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S

#function takes from JAXCosmo. Should likely be moved to an utils.py
def odeint(fn, y0, t):
    """Write documentation (TODO)."""

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
    Compute a linear interpolation (equivalent of np.interp).

    We are not doing any checks, so make sure your query points are lying
    inside the array.

    TODO: Implement proper interpolation, like in interpolations.jl

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
    r"""Compute a(z)."""
    return 1/(1+z)

#function from jaxcosmo
@jax.jit
def _romberg_diff(b, c, k):
    """
    Compute the differences for the Romberg quadrature corrections.

    See Forman Acton's "Real Computing Made Real," p 143.
    """
    tmp = 4.0**k
    return (tmp * c - b) / (tmp - 1.0)

#function from jaxcosmo
def romb(function, a, b, args=(), divmax=6, return_error=False):
    """
    Romberg integration of a callable function or method.

    Returns the integral of `function` (a function of one variable)
    over the interval (`a`, `b`).
    If `show` is 1, the triangular array of the intermediate results
    will be printed.  If `vec_func` is True (default is False), then
    `function` is assumed to support vector arguments.
    Parameters
    ----------
    function : callable
        Function to be integrated.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    Returns
    -------
    results  : float
        Result of the integration.
    Other Parameters
    ----------------
    args : tuple, optional
        Extra arguments to pass to function. Each element of `args` will
        be passed as a single argument to `func`. Default is to pass no
        extra arguments.
    divmax : int, optional
        Maximum order of extrapolation. Default is 10.
    See Also
    --------
    fixed_quad : Fixed-order Gaussian quadrature.
    quad : Adaptive quadrature using QUADPACK.
    dblquad : Double integrals.
    tplquad : Triple integrals.
    romb : Integrators for sampled data.
    simps : Integrators for sampled data.
    cumtrapz : Cumulative integration for sampled data.
    ode : ODE integrator.
    odeint : ODE integrator.
    References
    ----------
    .. [1] 'Romberg's method' http://en.wikipedia.org/wiki/Romberg%27s_method
    Examples
    --------
    Integrate a gaussian from 0 to 1 and compare to the error function.
    >>> from scipy import integrate
    >>> from scipy.special import erf
    >>> gaussian = lambda x: 1/np.sqrt(np.pi) * np.exp(-x**2)
    >>> result = integrate.romberg(gaussian, 0, 1, show=True)
    Romberg integration of <function vfunc at ...> from [0, 1]
    ::
       Steps  StepSize  Results
           1  1.000000  0.385872
           2  0.500000  0.412631  0.421551
           4  0.250000  0.419184  0.421368  0.421356
           8  0.125000  0.420810  0.421352  0.421350  0.421350
          16  0.062500  0.421215  0.421350  0.421350  0.421350  0.421350
          32  0.031250  0.421317  0.421350  0.421350  0.421350  0.421350  0.421350
    The final result is 0.421350396475 after 33 function evaluations.
    >>> print("%g %g" % (2*result, erf(1)))
    0.842701 0.842701
    """
    vfunc = jax.jit(lambda x: function(x, *args))

    n = 1
    interval = [a, b]
    intrange = b - a
    ordsum = _difftrap1(vfunc, interval)
    result = intrange * ordsum
    state = np.repeat(np.atleast_1d(result), divmax + 1, axis=-1)
    err = np.inf

    def scan_fn(carry, y):
        x, k = carry
        x = _romberg_diff(y, x, k + 1)
        return (x, k + 1), x

    for i in range(1, divmax + 1):
        n = 2**i
        ordsum = ordsum + _difftrapn(vfunc, interval, n)

        x = intrange * ordsum / n
        _, new_state = jax.lax.scan(scan_fn, (x, 0), state[:-1])

        new_state = np.concatenate([np.atleast_1d(x), new_state])

        err = np.abs(state[i - 1] - new_state[i])
        state = new_state

    if return_error:
        return state[i], err
    else:
        return state[i]

def _difftrap1(function, interval):
    """
    Perform part of the trapezoidal rule to integrate a function.

    Assume that we had called difftrap with all lower powers-of-2
    starting with 1.  Calling difftrap only returns the summation
    of the new ordinates.  It does _not_ multiply by the width
    of the trapezoids.  This must be performed by the caller.
        'function' is the function to evaluate (must accept vector arguments).
        'interval' is a sequence with lower and upper limits
                   of integration.
        'numtraps' is the number of trapezoids to use (must be a
                   power-of-2).
    """
    return 0.5 * (function(interval[0]) + function(interval[1]))

def _difftrapn(function, interval, numtraps):
    """
    Perform part of the trapezoidal rule to integrate a function.

    Assume that we had called difftrap with all lower powers-of-2
    starting with 1.  Calling difftrap only returns the summation
    of the new ordinates.  It does _not_ multiply by the width
    of the trapezoids.  This must be performed by the caller.
        'function' is the function to evaluate (must accept vector arguments).
        'interval' is a sequence with lower and upper limits
                   of integration.
        'numtraps' is the number of trapezoids to use (must be a
                   power-of-2).
    """
    numtosum = numtraps // 2
    h = (1.0 * interval[1] - 1.0 * interval[0]) / numtosum
    lox = interval[0] + 0.5 * h
    points = lox + h * np.arange(0, numtosum)
    s = np.sum(function(points))
    return s

@jax.jit
def Pkl_interp(k_l, z_l, ks, zs, Pk):
    """Write documentation (TODO)."""
    return 10**interpax.interp2d(np.log10(k_l), z_l, np.log10(ks), zs, np.log10(Pk),
                                 method="cubic")

Pkl_interp_vmap = jax.jit(jax.vmap(Pkl_interp, in_axes=(0, None, None, None, None)))

#from 2410.14623
def As_to_sigma8_max_precision(As, Om, Ob, h, ns, mnu, w0, wa):
    """
    Compute the emulated conversion As -> sigma8, using the most accurate expression.

    Args:
        :As (float): 10^9 times the amplitude of the primordial P(k)
        :Om (float): The z=0 total matter density parameter, Om
        :Ob (float): The z=0 baryonic density parameter, Ob
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :mnu (float): Sum of neutrino masses [eV / c^2]
        :w0 (float): Time independent part of the dark energy EoS
        :wa (float): Time dependent part of the dark energy EoS

    Returns:
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    """
    b = np.array([0.0246, 2.1062, 2.9355, 0.7626, 0.2962, 0.5096,
                4.4025, 3.6495, 0.4144, 0.8615, 0.6188, 0.1751,
                0.824, 0.5466, 0.5519, 0.3689, 0.3261, 0.2002,
                0.8892, 0.4462, 1.215, 3.4829, 2.5852, 0.0242,
                0.0051, 0.1614, 1.2991, 4.1426, 3.3055, 0.5716,
                6.0094, 1.9569, 2.1477, 1.1902, 0.128, 0.6931,
                0.2661])

    term1_inner = (Om * b[1] +
                (b[2] * mnu - b[3] * ns + np.log(b[4] * h - b[5] * mnu)) *
                (b[6] * h + b[7] * mnu - b[8] * ns + 1))
    term1 = b[0] * term1_inner

    term2 = b[9] * h - mnu

    term3_inner1 = (b[12] * w0 - b[13] * wa - np.log(Om * b[14])) * \
                (Om * b[15] + b[16] * w0 + b[17] * wa + np.log(-b[18] * w0 - b[19] * wa))
    term3_inner2 = np.log(Om * b[20] + np.log(-b[21] * w0 - b[22] * wa))
    term3 = b[10] * w0 - b[11] * mnu - term3_inner1 - term3_inner2 + np.log(-b[23] * w0 - b[24] * wa)

    term4_inner1 = Ob * b[30] - b[31] * h - np.log(Om * b[32])
    term4_inner2 = Om * b[33] - b[34] * h - b[35] * mnu - b[36] * ns
    term4 = b[25] * mnu - np.sqrt(Ob) * b[26] - Ob * b[27] + Om * b[28] - b[29] * h + 1 + term4_inner1 * term4_inner2

    result = term1 * term2 * term3 * term4

    return result*np.sqrt(As*10**9)

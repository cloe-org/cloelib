# cloelib imports
from cloelib.auxiliary.units import SPEED_OF_LIGHT
from cloelib.cosmology.cosmology import Perturbations
from cloelib.auxiliary.math_utils import cached_stacked_simpson

# General imports
import jax.numpy as np  # type: ignore
from scipy.interpolate import RectBivariateSpline  # type: ignore
import jax  # type: ignore
import interpax  # type: ignore
import jax.lax as lx

from cloe.cosmo.cosmology import Cosmology
from scipy import integrate, interpolate
from scipy.stats import skewnorm
from scipy.special import gamma, eval_legendre, spherical_jn, erf, j0, j1
from scipy.integrate import simpson as simps
from scipy.integrate import romberg
from scipy.integrate import quad_vec
from astropy import units
from astropy.constants import G


"""

## Notes:

- Make sufficiently general to interface with CAMB keeping the structure by Cosmology
- Make Tracer a protocol

"""

# UNITS
c_0 = SPEED_OF_LIGHT / 1000  # Convert to km/s


class CG:
    r"""
    Class for clusters of galaxies likelihood observable
    """

    def __init__(
        self,
        cosmo_dic,
        profile="NFW",
        overdensity_type="vir",
        overdensity=None,
        two_hale=None,
        offcentering=False,
        rms_off=0.0,
        f_off=0.0,
        zs_max=2.0,
        mean_nz=0.4,
        sigma_nz=0.3,
        alpha_nz=0.4,
        mshear=0.0,
        halo_concentration=0.1,
        two_halo="None",
        zed_obs_edges=np.linspace(0.2, 1.8, 9),
        Lambda_obs_edges=np.array([20.0, 30.0, 45.0, 60.0, 500.0]),
        Rad_obs_edges=np.linspace(5.0, 100.0, 11),
        Lambda_obs_Cxi2_edges=np.array([20, 30, 500]),
        Rad_obs_Cxi2_edges=np.geomspace(20.0, 130.0, 31),
        zed_obs_Cxi2_edges=np.arange(0.2, 1.81, 0.4),
        alpha_cov_Cxi2=None,
        beta_cov_Cxi2=None,
        gama_cov_Cxi2=None,
    ):

        # dictionary with all the input setting
        self.theory = cosmo_dic

        self.h = self.theory["H0"] / 100.0

        ################### GENERAL SETTING ###################
        self.profile = "NFW"  # self.theory['obs_specifications']['CG']['profile']
        self.overdensity_type = overdensity_type  # self.theory['obs_specifications']['CG']['overdensity_type']
        self.overdensity = (
            overdensity  # self.theory['obs_specifications']['CG']['overdensity']
        )

        self.two_halo = two_halo  # self.theory['obs_specifications']['CG']['two_halo']
        if self.two_halo not in ["None", "sum", "max"]:
            raise ValueError("Invalid 'two_halo' definition, %s." % self.two_halo)
        ############################################################################################
        #### ADD THIS CHECK OR SOMETHING SIMILAR FOR EVERY OPTION WITH MULTIPLE DEFINED CHOICES ####
        ############################################################################################

        # area of the survey [deg2]
        self.area = self.theory["obs_specifications"]["CG"]["area"]

        # models for halo mass function, halo bias and neutrino paradigm
        self.hmf = 1  # currently we only implemented the Castro+22 hmf. If we don't implemet other models this setting is useless
        self.bias = self.theory["obs_specifications"]["CG"][
            "bias"
        ]  # Tinker10 or Castro23
        self.neutrino_cdm = self.theory["obs_specifications"]["CG"]["neutrino_cdm"]

        # WL settings
        self.offcentering = offcentering  # self.theory['obs_specifications']['CG']['offcentering']  # offcentering
        self.rms_off = rms_off  # self.theory['obs_specifications']['CG']['rms_off']                # rms_off
        self.f_off = f_off  # self.theory['obs_specifications']['CG']['f_off']                  # f_off
        self.trunc_fact = 3.0

        # ???
        self.zs_max = zs_max  # self.theory['obs_specifications']['CG']['zs_max']
        self.mean_nz = mean_nz  # self.theory['obs_specifications']['CG']['mean_nz']
        self.sigma_nz = sigma_nz  # self.theory['obs_specifications']['CG']['sigma_nz']
        self.alpha_nz = alpha_nz  # self.theory['obs_specifications']['CG']['alpha_nz']

        # ???
        self.mshear = mshear  # self.theory['obs_specifications']['CG']['mshear']
        self.halo_concentration = halo_concentration  # self.theory['obs_specifications']['CG']['halo_concentration']

        ###############################################################################################################
        ###############################################################################################################
        ######### CHECK UNITS TO BE CONSISTENT WITHIN THE CODE                                                #########
        ######### UNITS OF INTEGRATION VARIABLES DO NOT MATTER, UNITS OF BINS DO                              #########
        ######### FOR NOW, INPUTS AND OUTPUTS ARE WITH h UNITS (AS FOR THE CODE)                              #########
        ######### BUT TO BE CONSISTENT WITH GENERAL CLOE WE SHOULD KEEP ALL THE INPUTS AND OUTPUTS WITHOUT h, #########
        ######### AND THEN WORK AS WE WANT                                                                    #########
        ###############################################################################################################
        ###############################################################################################################

        ################### ARRAYS FOR INTEGRATION VARIABLES ###################

        # wavelength array (integration variable)
        k_div = 701
        k_min = 1e-4
        k_max = 5e1
        self.k = np.geomspace(k_min, k_max, k_div)

        # mass array (integration variable)
        M_min = 12.0  # in Msun h^-1
        M_max = 16.0
        self.M_div = 50
        self.Mass = np.logspace(M_min, M_max, self.M_div + 1)

        # true richness array (integration variable)
        Lambda_min = 5.0
        Lambda_max = 250.0
        self.Lambda_div = 50
        self.Lambda = np.geomspace(Lambda_min, Lambda_max, self.Lambda_div + 1)

        # true redshift array (integration variable)
        z_min = 1e-5
        z_max = (
            self.zs_max - 1e-5
        )  # correction needed for avoiding zero values in n_zs_norM computation
        self.z_div = 50
        self.zed = np.linspace(z_min, z_max, self.z_div + 1)

        # ??? evaluated at true redshift
        self.nzsnorM = np.vectorize(self.n_zs_norM)(self.zed)
        self.nzs = self.n_zs(self.zed)
        self.r_interp = np.logspace(-10, 2.5, 200)

        ################### OBSERVED BINS ###################

        # observed redshift bins for number counts and weak lensing
        self.zed_obs_edges = (
            zed_obs_edges  # self.theory['obs_specifications']['CG']['zed_obs_edges']
        )
        self.zed_obs_div = len(self.zed_obs_edges) - 1

        # observed richness bins for number counts and weak lensing
        self.Lambda_obs_edges = Lambda_obs_edges  # self.theory['obs_specifications']['CG']['Lambda_obs_edges']
        self.Lambda_obs_div = len(self.Lambda_obs_edges) - 1

        # observed radial separation bins for weak lensing
        self.Rad_obs_edges = (
            Rad_obs_edges  # self.theory['obs_specifications']['CG']['Rad_obs_edges']
        )
        self.Rad_obs_div = len(self.Rad_obs_edges) - 1

        # observed richness bins for clustering
        self.Lambda_obs_Cxi2_edges = Lambda_obs_Cxi2_edges  # self.theory['obs_specifications']['CG']['Lambda_obs_Cxi2_edges']
        self.Lambda_obs_Cxi2_div = len(self.Lambda_obs_Cxi2_edges) - 1

        # observed radial separation bins for clustering
        self.Rad_obs_Cxi2_edges = Rad_obs_Cxi2_edges  # self.theory['obs_specifications']['CG']['Rad_obs_Cxi2_edges']
        self.Rad_obs_Cxi2_div = len(self.Rad_obs_Cxi2_edges) - 1

        # observed redshift bins for clustering
        self.zed_obs_Cxi2_edges = zed_obs_Cxi2_edges  # self.theory['obs_specifications']['CG']['zed_obs_Cxi2_edges']
        self.zed_obs_Cxi2_div = len(self.zed_obs_Cxi2_edges) - 1

        ################### NUISANCE PARAMETERS ###################

        # Parameters defining the mean and scatter of richness-mass relation
        self.A_l = self.theory["nuisance_parameters"]["A_l"]
        self.B_l = self.theory["nuisance_parameters"]["B_l"]
        self.C_l = self.theory["nuisance_parameters"]["C_l"]

        self.sig_A_l = self.theory["nuisance_parameters"]["sig_A_l"]
        self.sig_B_l = self.theory["nuisance_parameters"]["sig_B_l"]
        self.sig_C_l = self.theory["nuisance_parameters"]["sig_C_l"]

        # Parameters defining the scatter of the observed-true richness PDF
        self.sig_lambda_norm = self.theory["nuisance_parameters"]["sig_lambda_norm"]
        self.sig_lambda_z = self.theory["nuisance_parameters"]["sig_lambda_z"]
        self.sig_lambda_exponent = self.theory["nuisance_parameters"][
            "sig_lambda_exponent"
        ]

        # Parameters defining the scatter of the observed-true redshift PDF
        self.sig_z_z = self.theory["nuisance_parameters"]["sig_z_z"]
        self.sig_z_lambda = self.theory["nuisance_parameters"]["sig_z_lambda"]

        # CLUSTERING covariance nuisance parameters
        # if self.theory['obs_specifications']['CG']['alpha_cov_Cxi2'] is not None:
        if alpha_cov_Cxi2 is not None:
            # to be fitted from simulations
            self.alpha_cov_Cxi2 = alpha_cov_Cxi2  # self.theory['obs_specifications']['CG']['alpha_cov_Cxi2']
            self.beta_cov_Cxi2 = beta_cov_Cxi2  # self.theory['obs_specifications']['CG']['beta_cov_Cxi2']
            self.gamma_cov_Cxi2 = gamma_cov_Cxi2  # self.theory['obs_specifications']['CG']['gamma_cov_Cxi2']
        else:
            # reference values (=no corrections to the analytical covariance)
            self.alpha_cov_Cxi2 = np.zeros(
                (self.zed_obs_Cxi2_div, self.Lambda_obs_Cxi2_div)
            )
            self.beta_cov_Cxi2 = np.ones(
                (self.zed_obs_Cxi2_div, self.Lambda_obs_Cxi2_div)
            )
            self.gamma_cov_Cxi2 = np.zeros(
                (self.zed_obs_Cxi2_div, self.Lambda_obs_Cxi2_div)
            )

        ################### SELECTION FUNCTION ###################

        # selections (???)
        self.l_m_tab_sig_Cxi2 = [31, 51]
        self.l_m_tab_sig = [31, 31, 31, 51]
        self.z_tab_sig = 31

        # read richness selection function from table and interpolate
        self.external_richness_selection_function = self.theory["obs_specifications"][
            "CG"
        ]["external_richness_selection_function"]

        if self.external_richness_selection_function == "CG_ESF":
            self.int_Plobltr_Dlob = []
            ltr_grid = 10.0 ** (np.linspace(np.log10(5.0), np.log10(100.0), 100))
            ztr_grid = np.linspace(0.1, 1.8, 18)

            for ltab in range(len(self.Lambda_obs_edges) - 1):
                self.int_Plobltr_Dlob.append(
                    interpolate.interp2d(
                        ztr_grid,
                        ltr_grid,
                        CG_ricH_seL_funcT[ltab, :, :],
                        kind="cubic",
                        fill_value=None,
                    )
                )

    def Pk_def(self, z, k, nu_cdm):
        r"""
        Computes the power spectrum for the clusters probe.


        Parameters
        ----------
        z: float
            Redshift at which to evaluate the power spectrum.
        k: float or list or numpy.ndarray
            Wavenumber at which to evaluate the  power spectrum.
            Units: h Mpc^{-1}

        Returns
        -------
            float or numpy.ndarray
            Value of power spectrum
            at a given redshift and k-mode for galaxy clusters
            Units: h^{-3} Mpc^3
        """

        # general cloe works without h units
        k_noh = k * self.h

        if nu_cdm == "cb":
            return (self.theory["Pk_cb"].P(z, k_noh)) * self.h**3
        else:
            return (self.theory["Pk_delta"].P(z, k_noh)) * self.h**3

    def dV_dzdO(self, z):
        r"""
        Computes the volume element per redshit per solid angle
        at the redshift requested


        Parameters
        ----------
        z: float or numpy.ndarray
                   redshift at which to evaluate dV_dzdO

        Returns
        -------
        dV_dzdO: numpy.ndarray
                dV_dzdO[i] where i is the redshift axis
                Units: Mpc^3 h^{-3}
        """

        return (
            self.theory["c"]
            * self.theory["r_z_func"](z) ** 2.0
            / self.theory["H_z_func"](z)
            * self.h**3.0
        )

    def rho_crit_0(self):
        r"""
        Critical density of the universe at redshift = 0
        Units: Mpc^{-3} Ms h^2
        """

        hubble_value = 100.0 / 3.085677581491367e19  # H0/h / unit conversion
        G_unit = G.to(units.Mpc**3.0 / (units.Msun * units.s**2)).value

        return 3.0 * hubble_value**2.0 / (8.0 * np.pi * G_unit)

    def rho_crit_z(self, z):
        r"""
        Critical density of the universe at a redshift
        Units: Mpc^{-3} Ms h^2
        """

        hubble_value = (
            self.theory["H_z_func"](z) / 3.085677581491367e19 / self.h
        )  # Hz/h / unit conversion
        G_unit = G.to(units.Mpc**3.0 / (units.Msun * units.s**2.0)).value

        return 3.0 * hubble_value**2.0 / (8.0 * np.pi * G_unit)

    def rho_mean_0(self):
        r"""
        Mean matter density at redshift=0
        Units: Mpc^{-3} Ms h^2
        """

        if self.neutrino_cdm == "cb":
            return (self.theory["Omm"] - self.theory["Omnu"]) * self.rho_crit_0()
        else:
            return self.theory["Omm"] * self.rho_crit_0()

    def radius_M(self, M):
        r"""
        Convert the requested mass in the associated radius_M

        Parameters
        ----------
        M: float or numpy.ndarray
              Mass at which the radius is
              to be estimated in h^{-1} Ms

        Returns
        -------
        radius_M: array
                Radius_M in h^{-1} Mpc
        """

        return (M / self.rho_mean_0() * (3.0 / (4.0 * np.pi))) ** (1 / 3.0)

    def Omm_z(self, z, nu_cdm):
        r"""
        Computes the evolution of the matter density
        parameter with reshift

        Parameters
        ----------
        z: float or numpy.ndarray
                  Redshifts at which the computation parameter is
                  to be estimated
        """

        if nu_cdm == "cb":
            return (
                (self.theory["Omm"] - self.theory["Omnu"])
                * (1.0 + z) ** 3.0
                * (self.theory["H0"] / self.theory["H_z_func"](z)) ** 2.0
            )
        else:
            return (
                self.theory["Omm"]
                * (1.0 + z) ** 3.0
                * (self.theory["H0"] / self.theory["H_z_func"](z)) ** 2.0
            )

    def rho_mean_z(self, z, nu_cdm):
        r"""
        Mean matter density at z
        Units: Mpc^{-3} Ms h^2
        """

        return self.Omm_z(z, nu_cdm) * self.rho_crit_z(z)

    def delta_c(self, z):
        r"""
        Computes the critical overdensity at a given redshift
        following an approximation from Kitayama & Suto (1999)

        Parameters
        ----------
        z: float
            Redshift at which to evaluate the delta_c

        Returns
        -------
        delta_c:  float or numpy.ndarray
            Value of the critical overdensity a given redshift
        """

        return (
            3.0
            / 20.0
            * (12.0 * np.pi) ** (2.0 / 3.0)
            * (1.0 + 0.012299 * np.log10(self.Omm_z(z, nu_cdm="tot")))
        )

    def window(self, k, R):
        r"""
        Computes the top-hat window function and its derivative

        Parameters
        ----------
        k: float or numpy.ndarray
               Wavenumber at which to evaluate W(kR)
               Units:  h Mpc^{-1}
        R: float or numpy.ndarray
               Radius at which to evaluate W(kR)
               Units: h^{-1} Mpc

        Returns
        -------
        W:   numpy.ndarray
             W[i,j] where i is the wavenumber axis and
                j the radius axis
        dWdx: numpy.ndarray
              dWdx[i,j] where i is the wavenumber axis and
                j the radius axis
        """

        x = R[:, np.newaxis] * k
        W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3.0
        dWdx = 3.0 * (np.sin(x) * (x**2.0 - 3.0) + 3.0 * x * np.cos(x)) / x**4.0

        return W, dWdx

    def sigma_z_M(self, z, M):
        r"""
        Computes the rms at the masses requested from
        the table given by the Boltzman code

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate sigma_z_M
        M: float or numpy.ndarray
               Mass at which to evaluate sigma_z_M in h^{-1} Mpc

        Returns
        -------
        sigma_z_M: numpy.ndarray
                sigma_z_M[i,j] where i is the redshift axis and
                j the mass axis
        """

        k = self.k  # h/Mpc
        R = self.radius_M(M)  # Mpc/h
        W, dWdx = self.window(k, R)
        return np.sqrt(
            (
                1
                / (2.0 * np.pi**2)
                * simps(
                    (k**2.0).reshape(1, 1, len(k))
                    * self.Pk_def(z, k, self.neutrino_cdm).reshape(len(z), 1, len(k))
                    * (W**2.0).reshape(1, len(R), len(k)),
                    k,
                    axis=-1,
                )
            )
        )

    def nu_z_M(self, z, M):
        r"""
        Computes the critical overdensity over the rms
        delta_c/sigma at a given redshift and mass

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate nu_z_M
        M: float or numpy.ndarray
               Mass at which to evaluate nu_z_M

        Returns
        -------
        nu_z_M:   numpy.ndarray
            nu_z_M[i,j] where i is the redshift axis and
                j the mass axis
        """

        return self.delta_c(z)[:, np.newaxis] / self.sigma_z_M(z, M)

    def dlns_dlnR(self, z, M):
        r"""
        Computes the derivatives of the log rms
        with respect to the radius
        at the redshift and mass requested

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate dlns_dlnR
        M: float or numpy.ndarray
               Mass at which to evaluate dlns_dlnR
               Units: Ms h^{-1}

        Returns
        -------
        dlns_dlnR: numpy.ndarray
                dlns_dlnR[i,j] where i is the redshift axis and
                j the mass axis
        """

        k = self.k  # h/Mpc
        R = self.radius_M(M)  # Mpc/h
        W, dWdx = self.window(k, R)
        dsigma2_dR = np.pi**-2 * simps(
            k.reshape(1, 1, len(k)) ** 3
            * self.Pk_def(z, k, self.neutrino_cdm).reshape(len(z), 1, len(k))
            * W.reshape(1, len(R), len(k))
            * dWdx.reshape(1, len(R), len(k)),
            k,
            axis=-1,
        )

        return R / (2 * self.sigma_z_M(z, M) ** 2) * dsigma2_dR

    def f_sigma_nu(self, z, M):
        r"""
        Computes the multiplicity function
        at the redshift and mass requested
        Computation of the Multiplicity function

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate f_sigma_nu
        M: float or numpy.ndarray
               Mass at which to evaluate f_sigma_nu in h^{-1} Ms

        Returns
        -------
        f_sigma_nu: numpy.ndarray
                f_sigma_nu[i,j] where i is the redshift axis and
                j the mass axis
        """

        a1 = 0.7962
        a2 = 0.1449
        az = -0.0658
        p1 = -0.5612
        p2 = -0.4743
        q1 = 0.3688
        q2 = -0.2804
        qz = 0.0251

        dlnsigmadlnR = self.dlns_dlnR(z, M)
        Ommz = self.Omm_z(z, self.neutrino_cdm)[:, np.newaxis]
        nu = self.nu_z_M(z, M)

        aR = a1 + a2 * (dlnsigmadlnR + 0.6125) ** 2.0
        a = aR * Ommz**az
        p = p1 + p2 * (dlnsigmadlnR + 0.5)
        qR = q1 + q2 * (dlnsigmadlnR + 0.5)
        q = qR * Ommz**qz
        A = 1.0 / (
            2.0 ** (-0.5 - p + q / 2.0)
            / np.sqrt(np.pi)
            * (2.0**p * gamma(q / 2.0) + gamma(-p + q / 2.0))
        )

        return (
            A
            * np.sqrt(2.0 * a / (np.pi))
            * np.exp(-a * nu**2.0 / 2.0)
            * (1.0 + 1.0 / (a * nu**2.0) ** p)
            * (nu * np.sqrt(a)) ** (q - 1.0)
        ) * nu

    def dn_dm(self, z, M):
        r"""
        Computes the derivative of the number density
        at the redshift and mass requested


        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate dn_dm
        M: float or numpy.ndarray
               Mass at which to evaluate dn_dm
               Units: h^{-1} Ms

        Returns
        -------
        dn_dm: numpy.ndarray
                dn_dm[i,j] where i is the redshift axis and
                j the mass axis h^4 Mpc^{-3} Ms^{-1}
        """

        dlnsigmadlnR = self.dlns_dlnR(z, M)

        return self.rho_mean_0() / M**2.0 * self.f_sigma_nu(z, M) * dlnsigmadlnR / (-3)

    def lnlambda(self, z, M):
        r"""
        Computes the theoretical richness
        at the true redshift and mass requested


        Parameters
        ----------
        z: float or numpy.ndarray
                   True redshift at which to evaluate the theoretical richness
        M: float or numpy.ndarray
               Mass at which to evaluate the theoretical richness
               Units: h^{-1} Ms

        Returns
        -------
        lnlambda : numpy.ndarray
                lnlambda[i,j] where i is the true redhshift axis and
                                    j the mass axis
        """

        # 3e14 is in [Ms h^-1] units
        return (
            np.log(self.A_l)
            + self.B_l * np.log(M / (3.0e14))
            + self.C_l * np.log((1.0 + z[:, np.newaxis]) / (1.0 + 0.45))
        )

    def scatter_lnl(self, z, M):
        r"""
        Computes the scatter of the theoretical richness probability distribution
        at the true redshift and mass requested


        Parameters
        ----------
        z: float or numpy.ndarray
                   True redshift at which to evaluate the theoretical richness scatter
        M: float or numpy.ndarray
               Mass at which to evaluate the theoretical richness scatter
               Units: h^{-1} Ms

        Returns
        -------
        scatter_lnl : float or numpy.ndarray
                scatter_lnl[i,j] where i is the true redhshift axis and
                                       j the mass axis
        """

        return (
            self.sig_A_l
            + self.sig_B_l * np.log(M / (3.0e14))
            + self.sig_C_l * np.log((1.0 + z[:, np.newaxis]) / (1.0 + 0.45))
        )

    def P_lnlbd(self, z, M, Lambda):
        r"""
        Computes the theoretical richness probability distribution
        at the Mass, true redshift and theoretical richness requested


        Parameters
        ----------
        z: float or numpy.ndarray
                   True redshift at which to evaluate the theoretical richness scatter
        M: float or numpy.ndarray
               Mass at which to evaluate the theoretical richness scatter
               Units: Ms h^{-1}
        Lambda: float or numpy.ndarray
               Theoretical richness at which to evaluate the theoretical richness

        Returns
        -------
        P_lnlbd: float or numpy.ndarray
                P_lnlbd[i,j,k] where i is the redshift,
                                     j is the mass,
                                     k is the observed richness
        """

        lnlambda1 = self.lnlambda(z, M)[:, :, np.newaxis]
        sigmalnl = self.scatter_lnl(z, M)[:, :, np.newaxis]
        Lambda = Lambda[np.newaxis, np.newaxis, :]

        return (
            1.0
            / (Lambda * np.sqrt(2.0 * np.pi * sigmalnl**2.0))
            * np.exp(-((np.log(Lambda) - lnlambda1) ** 2.0) / (2.0 * sigmalnl**2.0))
        )

    def scatter_lbdobs_lbd(self, z, lbd):
        r"""
        Computes the scatter of the observed richness probability distribution
        at the true redshift and theoretical richness requested


        Parameters
        ----------
        z: float or numpy.ndarray
                   True redshift at which to evaluate the scatter of the observed richness probability distribution
        lbd: float or numpy.ndarray
               Theoretical richness at which to evaluate the scatter of the observed richness probability distribution

        Returns
        -------
        scatter_lbobs_lbdz: float or numpy.ndarray
                scatter_lbobs_lbdz[i,j] where i is the redshift axis
                                              j is the theoretical richness axis
        """

        return (
            self.sig_lambda_norm + self.sig_lambda_z * z[:, np.newaxis]
        ) * lbd**self.sig_lambda_exponent

    def P_lbdobs_lbd(self, z, lbd, lbd_obs):
        r"""
        Computes the observed richness probability distribution
        at the theoretical richness and the true redshift and observed richness requested


        Parameters
        ----------
        z: float or numpy.ndarray
               True redshift at which to evaluate the observed richness probability distribution
        lbd: float or numpy.ndarray
                   Theoretical richness at which to evaluate the observed richness probability distribution
        lbd_obs: float or numpy.ndarray
               Observed richness at which to evaluate the observed richness probability distribution

        Returns
        -------
        P_lbdobs_lbd: float or numpy.ndarray
                P_lbdobs_lbd[i,j,k] where i is the redshift axis
                                          j is the the theoretical richness axis,
                                          k is the observed richness
        """

        sigma_lbdobslbd = self.scatter_lbdobs_lbd(z, lbd)[:, :, np.newaxis]

        return (
            1.0
            / (np.sqrt(2.0 * np.pi * sigma_lbdobslbd**2.0))
            * np.exp(
                -(
                    (
                        lbd_obs[np.newaxis, np.newaxis, :]
                        - lbd[np.newaxis, :, np.newaxis]
                    )
                    ** 2.0
                )
                / (2.0 * sigma_lbdobslbd**2.0)
            )
        )

    def scatter_zobs_z(self, lbd_obs, z):
        r"""
        Computes the scatter of the observed redshift probability distribution
        at the true redshift and observed richness requested


        Parameters
        ----------
        z: float or numpy.ndarray
                   True redshift at which to evaluate scatter_zobs_z
        lbd: float or numpy.ndarray
               Observed richness at which to evaluate scatter_zobs_z

        Returns
        -------
        scatter_zobs_z: float or numpy.ndarray
                scatter_zobs_z[i,j] where i is the true redshift axis
                                          j the observed richness axis
        """

        return self.sig_z_z * z + self.sig_z_lambda * lbd_obs

    def P_zobs_z(self, zed_obs, lbd_obs, z):
        r"""
        Computes the observed redshift probability distribution
        at the theoretical richness and the true redshift and observed redshift requested


        Parameters
        ----------
        z: float or numpy.ndarray
               True redshift at which to evaluate the observed redshift probability distribution
        lbd: float or numpy.ndarray
                   Observed richness at which to evaluate the observed redshift probability distribution
        zed_obs: float or numpy.ndarray
               Observed redshift at which to evaluate the observed redshift probability distribution

        Returns
        -------
        P_zobs_z: float or numpy.ndarray
                P_zobs_z[i,j,k] where i is the observed redshift axis
                                      j is the observed richness axis
                                      k is the true redshift axis
        """

        sigmazobsz = self.scatter_zobs_z(lbd_obs, z)

        return (
            1.0
            / (np.sqrt(2.0 * np.pi * sigmazobsz**2.0))
            * np.exp(-((zed_obs[:, np.newaxis] - z) ** 2.0) / (2.0 * sigmazobsz**2.0))
        )

    def sigma_crit(self, z, z_sources):
        r"""
        Computes the critical surface mass density


        Parameters
        ----------
        z: float
                   Redshift at which to evaluate the critical density
        z_sources: float
                   Redshift of the galaxy sources

        Returns
        -------
        sigma_crit : float
                     Critical surface mass density (unit : Msun/pc^2)
        """

        light_speed = self.theory["c"] * (units.km / units.s)
        fact = light_speed**2.0 / (4.0 * np.pi * G)
        fact = fact.to(units.Msun / units.pc).value
        d_a_sources = self.theory["d_z_func"](z_sources) * 1.0e6
        d_a_l = self.theory["d_z_func"](z) * 1.0e6
        d_m_l = (1.0 + z) * d_a_l
        d_m_sources = (1.0 + z_sources) * d_a_sources
        d_h = self.theory["c"] / (self.theory["H0"] * 1.0e-6)
        d_a_lens_source = (
            1.0
            / (1.0 + z_sources)
            * (
                d_m_sources
                * np.sqrt(
                    1.0 + self.theory["Omk"] * (d_m_l[:, np.newaxis] ** 2.0 / d_h**2.0)
                )
                - d_m_l[:, np.newaxis]
                * np.sqrt(1.0 + self.theory["Omk"] * (d_m_sources**2.0 / d_h**2.0))
            )
        )
        sig_crit = fact * (d_a_sources / (d_a_l[:, np.newaxis] * d_a_lens_source))

        return sig_crit / self.h  # Ms pc^{-2} h

    def n_zs_norM(self, z):
        r"""
        galaxy number density normalization per redshift


        Parameters
        ----------
        z: float or np.ndarray
                   Redshift at which to evaluate the
                   normalization of the galaxy number density

        Returns
        -------
        n_zs_norM: float or np.ndarray
                   Galaxy number density normalization per redshift
        """

        n_zs_norM = 1.0 / (
            skewnorm.cdf(self.zs_max, self.alpha_nz, self.mean_nz, self.sigma_nz)
            - skewnorm.cdf(z, self.alpha_nz, self.mean_nz, self.sigma_nz)
        )

        return n_zs_norM

    def n_zs(self, z):
        r"""
        galaxy number density per redshift


        Parameters
        ----------
        z: float or np.ndarray
                   Redshift at which to evaluate the
                   galaxy number density

        Returns
        -------
        n_zs: float or np.ndarray
               Galaxy number density per redshift
        """

        n_zs = np.zeros((len(z), self.z_div + 1))
        for z_ind, zed in enumerate(z):
            z_s = np.linspace(zed + 1.0e-5, self.zs_max, self.z_div + 1)
            n_zs[z_ind] = skewnorm.pdf(z_s, self.alpha_nz, self.mean_nz, self.sigma_nz)

        return n_zs

    def m_sig_crit_m1(self, z, zbin):
        r"""
        Effective critical surface mass density


        Parameters
        ----------
        z: float
                Redshift at which to evaluate the
                effective critcal surface mass density

        Returns
        -------
        m_sigma_crit_m1: float
                Effective critical surface mass density (units : pc^2/Msun)
        """

        z_s = np.linspace(z + 1.0e-5, self.zs_max, self.z_div + 1, axis=1)
        sig_crit_m1 = self.nzs[zbin] * 1.0 / self.sigma_crit(z, z_s)

        return self.nzsnorM[zbin] * simps(sig_crit_m1, x=z_s)  # pc^2 / Msun / h

    def get_Delta(self, overdensity_type, z, nu_cdm, overdensity=200):
        r"""
        Overdensity factor.

        Parameters
        ----------
        overdensity_type: str
            The overdensity type. Possibilities are: "crit", "mean", "vir".
        z: float
            Redshift.
        overdensity: int
            The overdensity. If a virial density is assumed, this input
            variable is not used.

        Returns
        -------
        overdensity: float
            The overdensity factor which needs to be multiplied to the critical
            density in order to define an overdensity.

        Notes
        -----
        The function is returned for :math:`\rm \rho_c` in a density definition
        at a given redshift. The function returns :math:`\rm \Delta` for the
        critical density of the universe, :math:`\rm \Delta \Omega_{m}` for
        the mean matter density of the universe, :math:`\rm \Delta` determined
        by `Bryan & Norman 1998
        <http://adsabs.harvard.edu/abs/1998ApJ...495...80B>`_ Equation 6 for
        the virial density.
        """
        if overdensity_type == "crit":
            Delta = overdensity

        elif overdensity_type == "mean":
            Delta = overdensity * self.Omm_z(z, nu_cdm)

        elif overdensity_type == "vir":
            x = self.Omm_z(z, nu_cdm) - 1.0
            Delta = 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2

        else:
            raise ValueError("Invalid overdensity definition, %s." % overdensity_type)

        return Delta

    def F_NFW(self, x):
        r"""
        One-Halo NFW F term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        F_NFW: float
                One-Halo NFW F term.

        Notes
        -----
        Implementation of second part of Eq. 4 from `Golse et al. 2002
        <https://ui.adsabs.harvard.edu/abs/2002A%26A...390..821G/abstract>`_.
        """
        if x < 1.0:
            return (1.0 - np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)) / (x**2.0 - 1.0)
        if x == 1.0:
            return 1.0 / 3.0
        if x > 1.0:
            return (1.0 - np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)) / (x**2.0 - 1.0)

    def G_NFW(self, x):
        r"""
        One-Halo NFW G term.

        Parameters
        ----------
        x: float
            Dimensionless radial coordinates.

        Returns
        -------
        F_NFW: float
                One-Halo NFW G term.

        Notes
        -----
        Implementation of Eq. 5 from `Golse et al. 2002
        <https://ui.adsabs.harvard.edu/abs/2002A%26A...390..821G/abstract>`_.
        """
        if x < 1.0:
            return np.log(x / 2.0) + np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)
        if x == 1.0:
            return 1.0 + np.log(1.0 / 2.0)
        if x > 1.0:
            return np.log(x / 2.0) + np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)

    def F_BMO(self, x):
        r"""
        One-Halo BMO F term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        F_BMO : float
                One-Halo BMO F term.

        Notes
        -----
        Implementation of Eq. A.5 from `Baltz et al. 2009
        <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_.
        """
        if x < 1.0:
            return np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2.0)
        if x == 1.0:
            return 1.0
        if x > 1.0:
            return np.arccos(1.0 / x) / np.sqrt(x**2.0 - 1.0)

    def G_BMO(self, x):
        r"""
        One-Halo BMO G term.

        Parameters
        ----------
        x : float
            Dimensionless radial coordinates.

        Returns
        -------
        G_BMO : float
                One-Halo BMO G term.

        Notes
        -----
        Implementation of Eq. A.28 from `Baltz et al. 2009
        <https://ui.adsabs.harvard.edu/abs/2009JCAP...01..015B/abstract>`_.
        """
        if x < 1.0:
            return (self.F_BMO(x) - 1.0) / (1.0 - x**2.0)
        if x == 1.0:
            return 1.0 / 3.0
        if x > 1.0:
            return (1.0 - self.F_BMO(x)) / (x**2.0 - 1.0)

    def surface_mass_density_cen(self, R, z, c, M, bias="tinker10", force_no_2h=False):
        r"""
        Centered surface mass density profile at radius R

        Parameters
        ----------
        R: np.ndarray
            Radius at which the profile is to be computed (units : Mpc)
        z: float
            Redshift at which the mean matter contant is
            to be computed
        c: float
            Concentration parameter of the cluster
        M: Float
            Mass of the cluster (Msun)
        bias: float or string
            Halo bias. It can be a float or a string, and in
            the latter case the possibilities are: "tinker10"
        force_no_2h: bool
            if True, force the non-inclusion of the 2-halo term

        Returns
        -------
        surface_mass_density: np.ndarray
                              centered surface mass density profile (units : Msun / pc**2)

        """

        Delta_vir = self.get_Delta(
            self.overdensity_type, z[:, np.newaxis], "tot", self.overdensity
        )
        rho_c = self.rho_crit_z(z[:, np.newaxis])
        densityThreshold = Delta_vir * rho_c

        RDelta = (3.0 * M / 4.0 / np.pi / densityThreshold) ** (1.0 / 3.0)
        Rs = RDelta / c
        x = R / Rs

        if self.profile == "NFW":
            F = np.vectorize(self.F_NFW)(x)
            m_nfw = np.log(1.0 + c) - c / (1.0 + c)  # Eq. 4 Oguri & Hamana 2011
            rho_s = Delta_vir * c**3.0 / (3.0 * m_nfw) * rho_c

            Sigma = 2.0 * rho_s * Rs * F * 1.0e-12

        elif self.profile == "BMO":
            Rt = self.trunc_fact * RDelta
            tau = Rt / Rs

            m_bmo = (
                tau**2.0
                / (2.0 * (tau**2.0 + 1.0) ** 3.0 * (1.0 + c) * (tau**2.0 + c**2.0))
                * (
                    c
                    * (tau**2.0 + 1.0)
                    * (
                        c * (c + 1.0)
                        - tau**2.0 * (c - 1.0) * (2.0 + 3.0 * c)
                        - 2.0 * tau**4.0
                    )
                    + tau
                    * (c + 1.0)
                    * (tau**2.0 + c**2.0)
                    * (
                        2.0 * (3.0 * tau**2.0 - 1.0) * np.arctan(c / tau)
                        + tau
                        * (tau**2.0 - 3.0)
                        * np.log(tau**2.0 * (1.0 + c) ** 2.0 / (tau**2.0 + c**2.0))
                    )
                )
            )

            rho_s_bmo = Delta_vir * c**3.0 / (3.0 * m_bmo) * rho_c

            const = rho_s_bmo * Rs

            G = np.vectorize(self.G_BMO)(x)
            F = np.vectorize(self.F_BMO)(x)

            term1 = tau**4.0 / (tau**2.0 + 1.0) ** 3.0
            term2 = 2.0 * (tau**2.0 + 1.0) * G
            term3 = 8.0 * F
            term4 = (tau**4.0 - 1.0) / (tau**2.0 * (tau**2.0 + x**2.0))
            term5 = (
                np.pi
                * (4.0 * (tau**2.0 + x**2.0) + tau**2.0 + 1.0)
                / (tau**2.0 + x**2.0) ** (3.0 / 2.0)
            )
            term6 = (
                tau**2.0 * (tau**4.0 - 1.0)
                + (tau**2.0 + x**2.0) * (3.0 * tau**4.0 - 6.0 * tau**2.0 - 1.0)
            ) / (tau**3.0 * (tau**2.0 + x**2.0) ** (3.0 / 2.0))

            L = np.log(x / (np.sqrt(tau**2.0 + x**2.0) + tau))

            Sigma = 1e-12 * const * term1 * (term2 + term3 + term4 - term5 + term6 * L)

        else:
            raise ValueError("Invalid profile definition, %s." % self.profile)

        if force_no_2h == False and self.two_halo == "sum":
            Sigma += self.surface_mass_density_2h(R, z, bias, M)
        elif force_no_2h == False and self.two_halo == "max":
            Sigma_2h = self.surface_mass_density_2h(R, z, bias, M)
            Sigma = np.maximum(Sigma, Sigma_2h)

        return Sigma

    def surface_mass_density(
        self, R, z, c, M, bias="tinker10", force_no_2h=False, force_no_off=False
    ):
        r"""
        Surface mass density profile at radius R

        Parameters
        ----------
        R: np.ndarray
            Radius at which the profile is to be computed (units : Mpc)
        z: float
            Redshift at which the mean matter contant is
            to be computed
        c: float
            Concentration parameter of the cluster
        M: Float
            Mass of the cluster (Msun)
        bias: float or string
            Halo bias. It can be a float or a string, and in
            the latter case the possibilities are: "tinker10"
        force_no_2h: bool
            if True, force the non-inclusion of the 2-halo term
        force_no_off: bool
            if True, force the non-inclusion of the off-centering

        Returns
        -------
        surface_mass_density: np.ndarray
                              surface mass density profile (units : Msun / pc**2)

        """
        if force_no_off == False and self.offcentering and self.rms_off >= 1.0e-4:

            R = np.asarray(R)
            Sigma_off = np.zeros_like(R)

            ir.Sigma_off(
                R,
                self.r_interp,
                self.surface_mass_density_cen(
                    self.r_interp, z, c, M, bias, force_no_2h=False
                ),
                self.rms_off,
                Sigma_off,
            )

            Sigma_cen = self.surface_mass_density_cen(
                R, z, c, M, bias, force_no_2h=False
            )
            return (1.0 - self.f_off) * Sigma_cen + self.f_off * Sigma_off

        else:

            return self.surface_mass_density_cen(R, z, c, M, bias, force_no_2h)

    def excess_surface_mass_density(self, R, z, c, M, bias="tinker10"):
        r"""
        Excess surface mass density profile at radius R

        Parameters
        ----------
        R: np.ndarray
            Radius at which the profile is to be computed (units : Mpc)
        z: float
            Redshift at which the mean matter contant is
            to be computed
        c: float
            Concentration parameter of the cluster
        M: Float
            Mass of the cluster (Msun)
        bias: float or string
            Halo bias. It can be a float or a string, and in
            the latter case the possibilities are: "tinker10" and "castro23"

        Returns
        -------
        excess_surface_mass_density: float or np.ndarray
                                     excess surface density (units : Msun / pc**2)
        """
        Delta_vir = self.get_Delta(
            self.overdensity_type, z[:, np.newaxis], "tot", self.overdensity
        )
        rho_c = self.rho_crit_z(z[:, np.newaxis])
        densityThreshold = Delta_vir * rho_c

        RDelta = (3.0 * M / 4.0 / np.pi / densityThreshold) ** (1.0 / 3.0)
        Rs = RDelta / c
        x = R / Rs

        if self.profile == "NFW":
            G = np.vectorize(self.G_NFW)(x)

            m_nfw = np.log(1.0 + c) - c / (1.0 + c)  # Eq. 4 Oguri & Hamana 2011
            rho_s = Delta_vir * c**3.0 / (3.0 * m_nfw) * rho_c

            Sigma_mean = 4.0 * rho_s * Rs * (G / x**2.0) * 1.0e-12

        elif self.profile == "BMO":
            Rt = self.trunc_fact * RDelta
            tau = Rt / Rs

            m_bmo = (
                tau**2.0
                / (2.0 * (tau**2.0 + 1.0) ** 3.0 * (1.0 + c) * (tau**2.0 + c**2.0))
                * (
                    c
                    * (tau**2.0 + 1.0)
                    * (
                        c * (c + 1.0)
                        - tau**2.0 * (c - 1.0) * (2.0 + 3.0 * c)
                        - 2.0 * tau**4.0
                    )
                    + tau
                    * (c + 1.0)
                    * (tau**2.0 + c**2.0)
                    * (
                        2.0 * (3.0 * tau**2.0 - 1.0) * np.arctan(c / tau)
                        + tau
                        * (tau**2.0 - 3.0)
                        * np.log(tau**2.0 * (1.0 + c) ** 2.0 / (tau**2.0 + c**2.0))
                    )
                )
            )

            rho_s_bmo = Delta_vir * c**3.0 / (3.0 * m_bmo) * rho_c

            const = 2.0 * np.pi * rho_s_bmo * Rs**3.0
            term1 = tau**4.0 / (tau**2.0 + 1.0) ** 3.0

            F = np.vectorize(self.F_BMO)(x)
            term2 = 2.0 * (tau**2.0 + 1.0 + 4.0 * (x**2.0 - 1.0)) * F

            G = np.vectorize(self.G_BMO)(x)
            term3 = (
                np.pi * (3.0 * tau**2.0 - 1.0)
                + 2.0 * tau * (tau**2.0 - 3.0) * np.log(tau)
            ) / tau

            term4 = tau**3.0 * np.sqrt(tau**2.0 + x**2.0)
            term5 = -(tau**3.0) * np.pi * (4.0 * (tau**2.0 + x**2.0) - tau**2.0 - 1.0)
            term6 = -(tau**2.0) * (tau**4.0 - 1.0) + +(tau**2.0 + x**2.0) * (
                3.0 * tau**4.0 - 6.0 * tau**2.0 - 1.0
            )
            L = np.log(x / (np.sqrt(tau**2.0 + x**2.0) + tau))

            M_proj = const * term1 * (term2 + term3 + (term5 + term6 * L) / term4)

            Sigma_mean = M_proj / (np.pi * R**2.0) * 1.0e-12

        else:
            raise ValueError("Invalid profile definition, %s." % self.profile)

        Sigma = self.surface_mass_density_cen(R, z, c, M, bias, force_no_2h=True)
        DeltaSigma = Sigma_mean - Sigma

        if self.two_halo == "sum":
            DeltaSigma += self.excess_surface_mass_density_2h(R, z, bias, M)
        elif self.two_halo == "max":
            DeltaSigma_2h = self.excess_surface_mass_density_2h(R, z, bias, M)
            DeltaSigma = np.maximum(DeltaSigma, DeltaSigma_2h)

        if self.offcentering:

            if self.rms_off >= 1.0e-4 and self.f_off >= 1.0e-4:

                R = np.asarray(R)
                DeltaSigma_off = np.zeros_like(R)

                ir.DeltaSigma_off(
                    R,
                    self.r_interp,
                    self.r_interp,
                    self.surface_mass_density_cen(
                        self.r_interp, z, c, M, bias, force_no_2h=False
                    ),
                    self.rms_off,
                    DeltaSigma_off,
                )

                DeltaSigma *= 1.0 - self.f_off
                DeltaSigma += self.f_off * DeltaSigma_off

            elif self.rms_off < 1.0e-4 and self.f_off >= 1.0:

                DeltaSigma = np.zeros(len(DeltaSigma))

        return DeltaSigma

    def surface_mass_density_2h(self, R, z, bias, M=1e14):
        r"""
        Surface 2-halo density profile at radius R, generalized to handle arrays of z and M.

        Parameters
        ----------
        R: float
            Radius at which the profile is to be computed (units : Mpc)
        z: np.ndarray
            Redshift(s) at which the mean matter content is
            to be computed. Can be an array.
        bias: float or string
            Halo bias. It can be a float or a string, and in
            the latter case the possibilities are: "tinker10", "castro23"
        M: np.ndarray
            Mass(es) of the cluster(s) (Msun), used only for the bias computation.
            Can be an array.

        Returns
        -------
        surface_mass_density_2h: np.ndarray
            2-halo surface mass density profile (units : Msun / pc**2),
            computed for each z and M.
        """

        # Ensure z and M are arrays for broadcasting
        if type(z) is not np.ndarray:
            z = np.array([z])
        if type(M) is not np.ndarray:
            M = np.array([M])

        # Define base quantities
        D_A = self.theory["d_z_func"](z)  # D_A should now be (Nz, 1)
        theta = R / D_A  # R is a scalar, so theta has shape (Nz, 1)

        kl_min = 1.0e-4
        kl_max = 1.0e2
        kl_array = np.logspace(np.log10(kl_min), np.log10(kl_max), 500)

        # Bias calculation
        if isinstance(bias, str):
            if bias == "tinker10":
                Delta = self.get_Delta(
                    self.overdensity_type, z, "tot", self.overdensity
                ) / self.Omm_z(z, nu_cdm="tot")
                bias_z = self.bias_tinker10(z, M, Delta)  # (Nm,)
            elif bias == "castro23":
                bias_z = self.bias_castro23(z, M)  # (Nm,)
            else:
                raise ValueError(f"Invalid 'bias' definition: {bias}")
        else:
            # bias_z = np.full_like(M, bias)  # If bias is a scalar float, broadcast it
            bias_z = (
                np.ones((z.size, M.size)) * bias
            )  # If bias is a scalar float, broadcast it

        # Compute P(k) interpolation and Sigma for each redshift z
        Sigma = np.zeros((z.size, M.size))
        for i, z_val in enumerate(z):  # Loop over redshift values
            # Get P(k) for this redshift
            Pk_interp = interpolate.InterpolatedUnivariateSpline(
                kl_array, self.Pk_def(z_val, kl_array, nu_cdm="tot")
            )

            # Define the integrand for this redshift
            def integrand(l):
                kl = l / (1.0 + z_val) / D_A[i]
                return j0(l * theta[i]) * l * Pk_interp(kl)

            # Compute rho_m for this redshift
            rho_m = self.Omm_z(z_val, nu_cdm="tot") * self.rho_crit_z(z_val)

            # Compute Sigma for each mass M
            Sigma_z = quad_vec(
                integrand,
                kl_min * (1.0 + z_val) * D_A[i],
                kl_max * (1.0 + z_val) * D_A[i],
                epsrel=1e-1,
            )[0]

            Sigma_z *= (
                1.0e-12
                * rho_m
                * bias_z[i]
                / (2.0 * np.pi * (1.0 + z_val) ** 3.0 * D_A[i] ** 2.0)
            )

            # Store the result for this redshift
            Sigma[i, :] = Sigma_z

        return Sigma  # Shape: (Nz, Nm)

    def excess_surface_mass_density_2h(self, R, z, bias, M=1e14):
        r"""
        Excess surface 2-halo density profile at radius R

        Parameters
        ----------
        R: np.ndarray
            Radius at which the profile is to be computed (units : Mpc)
        z: float
            Redshift at which the mean matter contant is
            to be computed
        bias: float or string
            Halo bias. It can be a float or a string, and in
            the latter case the possibilities are: "tinker10" and "castro23"
        M: Float
            Mass of the cluster (Msun), used only for the bias computation

        Returns
        -------
        excess_surface_mass_density_2h: np.ndarray
                                 2-halo excess surface mass density profile (units : Msun / pc**2)

        """
        # Ensure z and M are arrays for broadcasting
        if type(z) is not np.ndarray:
            z = np.array([z])
        if type(M) is not np.ndarray:
            M = np.array([M])

        # Define base quantities
        D_A = self.theory["d_z_func"](z)  # D_A should now be (Nz, 1)
        theta = R / D_A  # R is a scalar, so theta has shape (Nz, 1)

        kl_min = 1.0e-4
        kl_max = 1.0e2
        kl_array = np.logspace(np.log10(kl_min), np.log10(kl_max), 500)

        # Bias calculation
        if isinstance(bias, str):
            if bias == "tinker10":
                Delta = self.get_Delta(
                    self.overdensity_type, z, "tot", self.overdensity
                ) / self.Omm_z(z, nu_cdm="tot")
                bias_z = self.bias_tinker10(z, M, Delta)  # (Nm,)
            elif bias == "castro23":
                bias_z = self.bias_castro23(z, M)  # (Nm,)
            else:
                raise ValueError(f"Invalid 'bias' definition: {bias}")
        else:
            # bias_z = np.full_like(M, bias)  # If bias is a scalar float, broadcast it
            bias_z = (
                np.ones((z.size, M.size)) * bias
            )  # If bias is a scalar float, broadcast it

        # Compute P(k) interpolation and Sigma for each redshift z
        DeltaSigma = np.zeros((z.size, M.size))
        for i, z_val in enumerate(z):  # Loop over redshift values
            # Get P(k) for this redshift
            Pk_interp = interpolate.InterpolatedUnivariateSpline(
                kl_array, self.Pk_def(z_val, kl_array, nu_cdm="tot")
            )

            # Define the integrand for this redshift
            def integrand(l):
                kl = l / (1.0 + z_val) / D_A[i]
                j2 = 2.0 / (l * theta[i]) * j1(l * theta[i]) - j0(l * theta[i])
                return j2 * l * Pk_interp(kl)

            # Compute rho_m for this redshift
            rho_m = self.Omm_z(z_val, nu_cdm="tot") * self.rho_crit_z(z_val)

            # Compute Sigma for each mass M
            DeltaSigma_z = quad_vec(
                integrand,
                kl_min * (1.0 + z_val) * D_A[i],
                kl_max * (1.0 + z_val) * D_A[i],
                epsrel=1e-1,
            )[0]
            DeltaSigma_z *= (
                1.0e-12
                * rho_m
                * bias_z[i]
                / (2.0 * np.pi * (1.0 + z_val) ** 3.0 * D_A[i] ** 2.0)
            )

            # Store the result for this redshift
            DeltaSigma[i, :] = DeltaSigma_z

        return DeltaSigma  # Shape: (Nz, Nm)

    def bias_tinker10(self, z, M, Delta):
        r"""
        Computes the Tinker+10 halo bias at a given redshift and mass

        Parameters
        ----------
        z: float or numpy.ndarray
                   Redshift at which to evaluate nu_z_M
        M: float or numpy.ndarray
               Mass at which to evaluate nu_z_M in h^{-1} Ms

        Returns
        -------
        bias:   numpy.ndarray
                Bias[i,j] where i is the redshift axis and
                j the mass axis
        """

        if type(M) is not np.ndarray:
            M = np.array([M])

        # parameters
        p = [1.0, 0.24, 0.44, 0.88, 0.183, 1.5, 0.019, 0.107, 0.19, 2.4]
        y = np.log10(Delta)
        A_par = p[0] + p[1] * y * np.e ** (-((4.0 / y) ** 4))
        a_par = p[2] * y - p[3]
        B_par = p[4]
        b_par = p[5]
        C_par = p[6] + p[7] * y + p[8] * np.e ** (-((4.0 / y) ** 4))
        c_par = p[9]

        # bias
        nu = self.nu_z_M(z, M).T
        return (
            1.0
            - A_par * nu**a_par / (nu**a_par + self.delta_c(z) ** a_par)
            + B_par * nu**b_par
            + C_par * nu**c_par
        ).T

    def bias_castro23(self, z, M):
        r"""
        Computes the Castro+23 halo bias at a given redshift and mass

        Parameters
        ----------
        z: float or numpy.ndarray
           Redshift at which to evaluate nu_z_M
        M: float or numpy.ndarray
           Mass at which to evaluate nu_z_M in h^{-1} Ms

        Returns
        -------
        bias: numpy.ndarray
              Bias[i,j] where i is the redshift axis and
              j the mass axis
        """

        # if the mass array has less than 4 entries, this causes problem with the derivative
        M = np.asarray(M)
        lenM_orig = M.size
        if lenM_orig < 4:
            M = np.append(M, M[-1] * np.arange(2, 6))

        dlnsigmadlnR = self.dlns_dlnR(z, M)
        Ommz = self.Omm_z(z, self.neutrino_cdm)[:, np.newaxis]
        S8 = self.theory["sigma8_0"] * np.sqrt(self.theory["Omm"] / 0.3)

        nu = self.nu_z_M(z, M)
        nufnu = self.f_sigma_nu(z, M)
        dlnnufnu_dlnnu = np.zeros(nufnu.shape)
        for i in range(len(z)):
            nufnu_int = interpolate.splrep(np.log(nu[i]), np.log(nufnu[i]), s=0)
            dlnnufnu_dlnnu[i] = interpolate.splev(np.log(nu[i]), nufnu_int, der=1)

        # parameters
        A0, a1, b1, b2, c1 = 1.150, 0.0929, 0.256, 0.173, -0.0372
        b_pbs = 1 - 1 / self.delta_c(z)[:, np.newaxis] * dlnnufnu_dlnnu
        f0 = 1 + a1 * Ommz
        f1 = 1 + b1 * dlnsigmadlnR + b2 * dlnsigmadlnR**2
        f2 = 1 + c1 * S8

        # bias
        bias = A0 * f0 * f1 * f2 * b_pbs

        # original mass array size
        if lenM_orig < len(M):
            bias = bias[:, :lenM_orig]

        return bias

    def Kl_coeff(self, L):
        r"""
        Coefficients of the spherical harmonics expansion of the angular part of the window function

        Parameters
        ----------
        L: int
           Maximum number at which to evaluate the coefficients

        Returns
        -------
        KL: numpy.ndarray
            Coefficients up to L multipole
        """

        ell = np.linspace(0, L, L + 1, dtype=int)
        theta = np.arccos(1 - (self.area * (np.pi / 180.0) ** 2.0) / (2 * np.pi))
        KL = (
            np.sqrt(np.pi / (2.0 * ell + 1.0))
            * (
                eval_legendre(ell - 1, np.cos(theta))
                - eval_legendre(ell + 1, np.cos(theta))
            )
            / (2.0 * np.pi * (1 - np.cos(theta)))
        )
        KL[0] = 1 / (2.0 * np.sqrt(np.pi))
        return KL

    def cov_window(self, zbin, ztab, k, L, KL):
        r"""
        Computes the window function between redshifts bins

        Parameters
        ----------
        zbins: int
               Index of the redshift bins at which to evaluate the window function
        ztab: numpy.ndarray
              Array of redshifts (integration variable) between zbins[zbin] and zbins[zbin+1]
        k: numpy.ndarray
           Wavenumbers used to evaluate power spectrum in h Mpc^{-1}
        L: int
           Maximum number at which the coefficients are evaluated
        KL: numpy.ndarray
            Spherical harmonic expansion coefficients

        Returns
        -------
        cluster count covariance window:   numpy.ndarray
                W[i,j,k] where i and j are two redshift bin and k are the wavenumbers
        """

        rvec = self.theory["r_z_func"](ztab) * self.h  # Mpc  h^{-1}
        Vz = (rvec[-1] ** 3 - rvec[0] ** 3) / 3  # Mpc^3 h^{-3}
        kr = self.k[:, np.newaxis] * rvec
        self.rint[zbin] = (
            1
            / Vz
            * simps(
                rvec**2.0
                * np.array(
                    [spherical_jn(l, kr, derivative=False) for l in range(L + 1)]
                ),
                rvec,
                axis=-1,
            ).T
        )

        return (4 * np.pi) * np.sum(
            self.rint[zbin, :, :] * self.rint[: (zbin + 1), :, :] * KL[:] ** 2, axis=-1
        )

    def WF_ra(self, z_array, r_array, k_array):
        r"""
        Computes the window function and the volume of the spherical shells as a function of the radial separation

        Parameters
        ----------
        z_array: float or numpy.ndarray
                 Redshift at which apply the geometrical correction (Alcock-Paczynski effect)
        k_array: float or numpy.ndarray
                 Wavenumber used to evaluate power spectrum
                 Units: h Mpc^{-1}
        r_array: float or numpy.ndarray
                 Radial separation bins
        Returns
        -------
        cluster count covariance window:   numpy.ndarray
                W[i,j,k] where i is the redshift bin, j is the radial bin and k are the wavenumbers
        spherical shell volume: numpy.ndarray
                V[i,j] where i is the redshift bin and j is the radial bin
        """
        z_array = z_array[:, np.newaxis, np.newaxis]
        r_array = r_array[np.newaxis, :, np.newaxis]
        k_array = k_array[np.newaxis, np.newaxis, :]

        r_z_array = (
            self.Dv_rs_func(z_array) * r_array
        )  # AP correction (adds a redshift dependence)
        r3_TH_filter = (
            r_z_array**3
            * 3.0
            * (
                np.sin(k_array * r_z_array)
                - k_array * r_z_array * np.cos(k_array * r_z_array)
            )
            / (k_array * r_z_array) ** 3.0
        )

        W_rad = (r3_TH_filter[:, 1:, :] - r3_TH_filter[:, :-1, :]) / (
            r_z_array[:, 1:, :] ** 3 - r_z_array[:, :-1, :] ** 3
        )

        V_rad = (
            4.0
            * np.pi
            / 3.0
            * ((r_z_array[:, 1:, 0]) ** 3 - (r_z_array[:, :-1, 0]) ** 3)
        )

        return W_rad, V_rad

    # cosmo correction
    def Dv_rs_func(self, z):
        """
        Compute the correction that accounts for the wrong cosmology assumed in the measurement of the 2ptCF
        See https://arxiv.org/pdf/1511.00012.pdf (Sect. 4.3.1) for details.

        Parameters
        ----------
        z: redshift

        Returns
        -------
        DV_over_rs: volume distance over drag scale (sound horizon scale at recombination)

        """

        # here we can work in Mpc (without h unit conversion), as the output is dimensionless

        # isotropic volume distance
        Dv = (
            (1 + z) ** 2
            * self.theory["d_z_func"](z) ** 2
            * self.theory["c"]
            * z
            / self.theory["H_z_func"](z)
        ) ** (1 / 3.0)

        # isotropic volume distance at fiducial cosmology (assumed for measuring the 2pcf)
        Dv_fid = (
            (1 + z) ** 2
            * self.theory["fid_d_z_func"](z) ** 2
            * self.theory["c"]
            * z
            / self.theory["fid_H_z_func"](z)
        ) ** (1 / 3.0)

        return (Dv / self.theory["rdrag"]) * (self.theory["fid_rdrag"] / Dv_fid)

    def photoz_rsd_correction(self, z, Lambda):
        """
        Compute the correction that accounts for photo-z uncertainty and RSD

        Parameters
        ----------
        z:  float or numpy.ndarray
            redshift
        Lambda: float
                richness

        Returns
        -------
        corr0, corr1, corr2: correction terms to the power spectrum

        """

        # growth rate
        f_gr = (self.Omm_z(z, self.neutrino_cdm) ** 0.55)[:, np.newaxis]

        sigma_zob = self.scatter_zobs_z(Lambda, z)
        ks = self.k * (
            sigma_zob * self.theory["c"] / self.theory["H_z_func"](z) * self.h
        ).reshape(len(z), 1)
        erf_ks = erf(ks)

        corr0 = np.sqrt(np.pi) / (2 * ks) * erf_ks
        corr1 = f_gr / ks**3 * (np.sqrt(np.pi) / 2 * erf_ks - ks * np.exp(-(ks**2)))
        corr2 = (
            f_gr**2
            / ks**5
            * (
                3 * np.sqrt(np.pi) / 8 * erf_ks
                - ks / 4 * (2 * ks**2 + 3) * np.exp(-(ks**2))
            )
        )

        # correct for numerical inaccuracy
        corr1[erf_ks < 0.02] = 2 / 3.0
        corr2[erf_ks < 0.02] = 1 / 5.0

        return corr0, corr1, corr2


def simps(f, a, b, N=128):
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S

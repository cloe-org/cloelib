from cloelib.cosmology.camb_cosmology import Perturbations

from ...auxiliary import units
from scipy.special import erf
import jax.numpy as np

class HaloClustering:
    def __init__(
        self,
        pertrurbations: Perturbations,
        pertrurbations_fid: Perturbations,
    ):

        self.background = pertrurbations.background
        self.background_fid = pertrurbations_fid.background


        
    def WF_ra(self, z: np.ndarray, r: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the window function and the volume of the spherical shells as a function of the radial separation

        Parameters
        ----------
        z: np.ndarray
           Redshift at which apply the geometrical correction (Alcock-Paczynski effect)
        k: np.ndarray
           Wavenumber used to evaluate power spectrum, in h Mpc^{-1}
        r: np.ndarray
           Radial separation bins in h^{-1} Mpc

        Returns
        -------
        cluster count covariance window:   numpy.ndarray
                W[i,j,k] where i is the redshift bin, j is the radial bin and k are the wavenumbers
        spherical shell volume: numpy.ndarray
                V[i,j] where i is the redshift bin and j is the radial bin
        """
        z = z[:, np.newaxis, np.newaxis]
        r = r[np.newaxis, :, np.newaxis]
        k = k[np.newaxis, np.newaxis, :]

        r_z = (
            self.APcorr_func(z) * r
        )  # AP correction (adds a redshift dependence)

        r3_TH_filter = (
            r_z**3
            * 3.0
            * (
                np.sin(k * r_z)
                - k * r_z * np.cos(k * r_z)
            )
            / (k * r_z) ** 3.0
        )

        W_rad = (r3_TH_filter[:, 1:, :] - r3_TH_filter[:, :-1, :]) / (
            r_z[:, 1:, :] ** 3 - r_z[:, :-1, :] ** 3
        )

        V_rad = (
            4.0
            * np.pi
            / 3.0
            * ((r_z[:, 1:, 0]) ** 3 - (r_z[:, :-1, 0]) ** 3)
        )

        return W_rad, V_rad


    # cosmo correction (isotropic AP)
    def APcorr_func(self,  z: np.ndarray) -> np.ndarray:
        """
        Compute the correction that accounts for the wrong cosmology assumed in the measurement of the 2ptCF
        See https://arxiv.org/pdf/1511.00012.pdf (Sect. 4.3.1) for details.

        Parameters
        ----------
        z: np.ndarray
           Redshift

        Returns
        -------
        AP_corr: np.ndarray
           Volume distance over drag scale (sound horizon scale at recombination) over the same quantity at fiducial cosmology

        """

        # isotropic volume distance
        Dv = (
            (1 + z) ** 2
            * self.background.angular_diameter_distance(z) ** 2
            * units.SPEED_OF_LIGHT
            * z
            / self.background.hubble_parameter(z)
        ) ** (1 / 3.0)

        # isotropic volume distance at fiducial cosmology (assumed for measuring the 2pcf)
        Dv_fid = (
            (1 + z) ** 2
            * self.background_fid.angular_diameter_distance(z) ** 2
            * units.SPEED_OF_LIGHT
            * z
            / self.background_fid.hubble_parameter(z)
        ) ** (1 / 3.0)

        return (Dv / self.background.rdrag) * (self.background_fid.rdrag / Dv_fid)



    # IR resummation of the bao wiggles in the Pk
    def Pk_IR_func(self, k: np.ndarray, Pk: np.ndarray) -> np.ndarray:
        """
        Infrared resummation (first order approx) to correct non-linear damping of bao wiggles

        Parameters
        ----------                                                                                                                                                                                                   
        k: np.ndarray                                                                                                                                                                                                
           Wavenumber in h/Mpc
        Pk: np.ndarray
           Linear matter power spectrum at different redshifts in (Mpc/h)^3
                                                                                                                                                                                                                     
        Returns                                                                                                                                                                                                      
        -------                                                                                                                                                                                                      
        Pk_IR: np.ndarray                                                                                                                                                                                          
           Matter power spectrum with corrected bao wiggles in (Mpc/h)^3
                                                                                                                                                                                                                     
        """
        
        ns   = self.background.ns
        h    = self.background.h
        Obh2 = self.background.Omega_b * h**2
        Omh2 = self.background.Omega_m(0., self.nonu) * h**2
        Tcmb =  2.73
        
        k    *= h  #  1/Mpc
        s     = 44.5 * np.log(9.83/Omh2) / np.sqrt(1.+10.*(Obh2)**0.75)
        Gamma = Omh2 / h
        AG    = (1. - 0.328 * np.log(431.*Omh2) * Obh2 / Omh2 + 0.38 * np.log(22.3 * Omh2) * (Obh2 / Omh2)**2)
        Gamma = Gamma * (AG + (1.-AG) / (1.+(0.43*k*s)**4))
        Theta = Tcmb / 2.7
        q     = k * Theta**2 / Gamma / h
        L0    = np.log(2.*np.e + 1.8*q)
        C0    = 14.2 + 731. / (1. + 62.5 * q)
        T0    = L0 / (L0 + C0 * q * q)
        T0   /= T0[0]
        P_EH  = k**ns * T0**2
        P_EH  = (P_EH[:,None] * (Pk[:,0] / P_EH[0])).T
        k    /= h  # h/Mpc again

        lamb  = 0.25
        kS    = 0.2 
        lOsc  = 102.707
        qlog  = np.log10(k)
        dqlog = qlog[1]-qlog[0]

        # Gaussian filtering for Pnw and Wiggle-smooth split                                                                                                                                                     
        Pnw = P_EH*np.array([dqlog/np.sqrt(2.*np.pi*lamb**2) *
                             (np.sum(np.e**(-0.5* (klog - qlog[(abs(qlog-klog)<4.*lamb)])**2/ lamb**2)
                                     * Pk[:,(abs(qlog-klog)<4.*lamb)]
                                     /P_EH[:,((abs(qlog-klog)<4.*lamb))],axis=1)) for klog in qlog]).T

        Pw  = Pk - Pnw

        #Sigma2 as integral up to 0.2                                                                                                                                                                              
        icut   = (k <= kS)
        kcut   = k[icut]
        Pnwcut = Pnw[:,icut]
        kosc   = 1./lOsc
        norm   = 1./(6.*np.pi**2)
        Sigma2 = norm*simps(Pnwcut*(1.-spherical_jn(0,kcut/kosc)+2.*spherical_jn(2,kcut/kosc)), x=kcut)

        # comput final power spectrum                                                                                                                                                                               
        Pk_IR = Pnw + np.e**(-k**2 * Sigma2[:,None]) *Pw

        return Pk_IR


    
    def photoz_rsd_correction(self, z: np.ndarray, sigma_zob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """                                                                                                                                                                                                          
        Compute the correction that accounts for photo-z uncertainty and RSD (Kaiser effect)                                                                                                              
                                                                                                                                                                                                                     
        Parameters                                                                                                                                                                                                   
        ----------                                                                                                                                                                                                   
        z:  np.ndarray
            redshift
        sigma_zob: np.ndarray
            photo-z uncertainty
        
        Returns                                                                                                                                                                                                      
        -------                                                                                                                                                                                                      
        corr0, corr1, corr2: np.ndarray, np.ndarray, np.ndarray
            Correction terms to the power spectrum monopole                                                                                                                                           
        """

        
        # growth rate                                                                                                                                                                                                
        ##f_gr = self.perturbations.growth_rate() **0.55  #redshift???
        f_gr = (self.cosmo.Omega_m(z, self.nonu) ** 0.55)[:, np.newaxis]

                
        ks = self.k * (
            sigma_zob * (units.SPEED_OF_LIGHT *1e-3) / self.background_fid.hubble_parameter(z)
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

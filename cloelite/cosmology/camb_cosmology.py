# cloelite imports
from cloelite.cosmology.cosmology import Background
from cloelite.cosmology.cosmology import LinearPerturbations
from cloelite.cosmology.cosmology import NonLinearPerturbations
# General imports
import numpy as np 
# Cosmology imports
try:
    import camb
    from camb import model
except ImportError:
    raise ImportError("camb could not be imported.")


"""

## Notes:

- Make sufficiently general to interface with CAMB keeping the structure by Cosmology

"""

class CAMBBackground(Background):
    def __init__(self, H0: float, Omb: float, Omc: float, Omk: float, sigma8: float, ns: float,
                 As: float, w: float, wa: float, gamma_MG: float):
        r"""
        A class to define background cosmology using CAMB
        and inheriting from Cosmology parent class
        """
        
        super().__init__(H0, Omb, Omc, Omk, sigma8, As, ns, w, wa, gamma_MG)
        self.h = float(self.H0 / 100)
        self.ombh2 = float(self.Omb * self.h**2)
        self.omch2 = float(self.Omc * self.h**2)

        # Define CAMB params
        self.CAMBparams = camb.CAMBparams()
        # For the moment, ignore neutrinos
        self.CAMBparams.set_cosmology(H0=H0, ombh2=self.ombh2, omch2=self.omch2, mnu=0.0, 
                                      neutrino_hierarchy='degenerate', num_massive_neutrinos=0.0, YHe=0.2454 , nnu=0.0)
        self.CAMBparams.set_dark_energy(w=self.w, wa=self.wa) #re-set defaults
        self.CAMBparams.InitPower.set_params(As = self.As, 
                                                        ns = self.ns)
        # Get background cosmology
        self.CAMBresults = camb.get_background(self.CAMBparams)

        # Update attributes with derived parameters
        self.Omm = self.CAMBparams.omegam
        self.Omnu = self.CAMBparams.omeganu

    def hubble_parameter(self, zs, units = '1/Mpc') -> np.ndarray:
            r"""
            Retrieves the hubble parameter as
            a function of redshift

            .. math::
                H(z) = \sqrt

            Parameters
            ----------
            zs: numpy.ndarray
                Redshifts for the matter density
            units: str
                Used units to return H(z)
                Options are: km/s/Mpc and 1/Mpc

            Returns
            -------
            Hubble parameter: numpy.ndarray
                hubble parameter as a function of redshift

            """

            if units == '1/Mpc':
                return self.CAMBresults.h_of_z(zs)
            elif units == 'km/s/Mpc':
                return self.CAMBresults.hubble_parameter(zs)

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

        return self.CAMBresults.comoving_radial_distance(zs)

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

        return self.CAMBresults.angular_diameter_distance(zs)

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

        omegam_z = self.CAMBresults.get_Omega('cdm', z=zs) + \
            self.CAMBresults.get_Omega('baryon', z=zs) + \
            self.CAMBresults.get_Omega('neutrino', z=zs) + \
            self.CAMBresults.get_Omega('nu', z=zs)
        return self.CAMBresults.get_Omega('tot', z=zs)
    
    def transverse_comoving_distance(self, zs) -> np.ndarray:
        """
        Calculates the transverse comoving distance beetween two redshifts.

        Parameters:
        -----------
        zs : numpy.ndarray
            Redshifts at which to calculate the transverse comoving distance.

        Returns:
        --------
        np.ndarray
            The transverse comoving distance as a function of redshift.
        """
        c_0 = 2.99792458e5
        int_z1z2 = ((self.comoving_distance(z)[None, :] - self.comoving_distance(z)[:, None]) *
                    self.H0 / c_0)
        if self.Omk == 0.0:
            y_int = int_z1z2
        elif self.Omk > 0.0:
            y_int = (np.sinh(np.sqrt(self.Omk) * int_z1z2) /
                     np.sqrt(self.Omk))
        else:
            y_int = (np.sin(np.sqrt(-self.Omk) * int_z1z2) /
                     np.sqrt(-self.Omk))
        y_int *= (self.cosmo_dic['c'] / self.H0)

        return y_int
    
class CAMBLinearPerturbations(LinearPerturbations):
    def __init__(self, background : CAMBBackground, redshifts: np.ndarray):
        r"""
        A class to define perturbations cosmology using JAX
        and inheriting from Cosmology parent class

        """

        self.background = background
        self.background.CAMBparams.NonLinear = model.NonLinear_none
        self.background.CAMBparams.set_matter_power(redshifts=redshifts, 
                                                    kmax=50)
        self.CAMBdata = camb.get_results(self.background.CAMBparams)

    def linear_matter_power_spectrum(self, zs, ks, kmax: float, extrap_kmax: float):
        r"""Computes the linear matter power spectrum.

        Parameters
        ----------
        ks: numpy.ndarray
            Wave number in h Mpc^{-1}

        zs: numpy.ndarray
            Scale factor (def: 1.0)

        Returns
        -------
        pk: numpy.ndarray
            Linear matter power spectrum at the specified scale
            and redshift

        """
        
        #Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
        #Here for lensing we want the power spectrum of the Weyl potential.
        pk_linear = camb.get_matter_power_interpolator(self.background.CAMBparams, 
                                                nonlinear=False, hubble_units=False, k_hunit=False, 
                                                kmax=kmax, extrap_kmax = extrap_kmax,
                                                var1='delta_tot',var2='delta_tot', 
                                                zmax=zs)
        
        self.Pk_linear = pk_linear
        
        return pk_linear.P(zs, ks)
    
    def growth_factor(self, zs, ks, kmax: float, extrap_kmax: float) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.


        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth factor.
        ks : array_like
            Wavenumbers at which to calculate the growth factor.

        Returns:
        --------
        np.ndarray
            The growth factor as a function of redshift and wavenumber.
        """
        if hasattr(self, 'Pk_linear') and self.Pk_linear is not None:
            D_z_k = np.sqrt(self.Pk_linear.P(zs, ks) / self.Pk_linear.P(0.0, ks))
        else:
            self.linear_matter_power_spectrum(zs, ks)
            D_z_k = np.sqrt(self.Pk_linear.P(zs, ks) / self.Pk_linear.P(0.0, ks))

        return D_z_k

    def sigma8(self) -> np.ndarray:
        """
        Calculates sigma8 for given redshifts

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth rate.

        Returns:
        --------
        np.ndarray
            The sigma8 as a function of redshift 
        """
        
        # This could be catch
        s8 = np.array(self.CAMBdata.get_sigma8())
        return s8[::-1]

    def fsigma8(self) -> np.ndarray:
        """
        Calculates sigma8 for given redshifts

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth rate.

        Returns:
        --------
        np.ndarray
            The sigma8 as a function of redshift 
        """

        fs8 = np.array(self.CAMBdata.get_fsigma8())
        return fs8[::-1]

    def growth_rate(self) -> np.ndarray:
        """
        Calculates the growth rate for given redshifts and wavenumbers.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth rate.
        ks : array_like
            Wavenumbers at which to calculate the growth rate.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """
        
        return self.fsigma8()/self.sigma8()

class CAMBNonLinearPerturbations(NonLinearPerturbations):
    def __init__(self, linearperturbations : CAMBLinearPerturbations, redshifts: np.ndarray,
                  nonlinear_model = 'mead2020'):
        
        self.linearperturbations = linearperturbations
        self.background = linearperturbations.background
        self.background.CAMBparams.NonLinear = model.NonLinear_both
        self.background.CAMBparams.NonLinearModel.set_params(halofit_version=nonlinear_model)
        self.background.CAMBparams.set_matter_power(redshifts=redshifts, 
                                                    kmax=2)
        self.CAMBdata = camb.get_results(self.background.CAMBparams)
        
    def nonlinear_matter_power_spectrum(self, zs, ks, kmax: float, extrap_kmax: float):
        r"""Computes the linear matter power spectrum.

        Parameters
        ----------
        ks: numpy.ndarray
            Wave number in h Mpc^{-1}

        zs: numpy.ndarray
            Scale factor (def: 1.0)

        Returns
        -------
        pk: numpy.ndarray
            Linear matter power spectrum at the specified scale
            and redshift

        """
        
        #Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
        #Here for lensing we want the power spectrum of the Weyl potential.
        pk_nonlinear = camb.get_matter_power_interpolator(self.background.CAMBparams, 
                                                nonlinear=True, hubble_units=False, k_hunit=False, 
                                                kmax=kmax, extrap_kmax = extrap_kmax,
                                                var1='delta_tot',var2='delta_tot', 
                                                zmax=zs)
        
        self.Pk_nonlinear = pk_nonlinear

        return pk_nonlinear.P(zs, ks)
    
    def growth_factor(self, zs, ks, kmax: float, extrap_kmax: float) -> np.ndarray:
        """
        Calculates the growth factor for given redshifts and wavenumbers.

        .. math::
            D(z, k) =\sqrt{P_{\rm \delta\delta}(z, k)\
            /P_{\rm \delta\delta}(z=0, k)}\\

        and normalizes as for :math:`D(z)/D(0)`.


        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth factor.
        ks : array_like
            Wavenumbers at which to calculate the growth factor.

        Returns:
        --------
        np.ndarray
            The growth factor as a function of redshift and wavenumber.
        """
        if hasattr(self, 'Pk_nonlinear') and self.Pk_nonlinear is not None:
            D_z_k = np.sqrt(self.Pk_nonlinear.P(zs, ks) / self.Pk_nonlinear.P(0.0, ks))
        else:
            self.nonlinear_matter_power_spectrum(zs, ks, kmax=kmax, extrap_kmax=extrap_kmax)
            D_z_k = np.sqrt(self.Pk_nonlinear.P(zs, ks) / self.Pk_nonlinear.P(0.0, ks))

        return D_z_k

    def sigma8(self) -> np.ndarray:
        """
        Calculates sigma8 for given redshifts

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth rate.

        Returns:
        --------
        np.ndarray
            The sigma8 as a function of redshift 
        """
        
        s8 = np.array(self.CAMBdata.get_sigma8())
        return s8[::-1]


    def fsigma8(self) -> np.ndarray:
        """
        Calculates sigma8 for given redshifts

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth rate.

        Returns:
        --------
        np.ndarray
            The sigma8 as a function of redshift 
        """
        
        fs8 = np.array(self.CAMBdata.get_fsigma8())
        return fs8[::-1]

    def growth_rate(self) -> np.ndarray:
        """
        Calculates the growth rate for given redshifts and wavenumbers.

        Parameters:
        -----------
        zs : array_like
            Redshifts at which to calculate the growth rate.
        ks : array_like
            Wavenumbers at which to calculate the growth rate.

        Returns:
        --------
        np.ndarray
            The growth rate as a function of redshift and wavenumber.
        """
        
        return self.fsigma8()/self.sigma8()

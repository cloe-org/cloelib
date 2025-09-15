# cloelib imports
from cloelib.observables.clusters.halo_statistics import HaloStatistics, HaloStatisticsCastro, HaloStatisticsTinker
from cloelib.observables.clusters.selection_function import SelectionFunction
from cloelib.cosmology.cosmology import Perturbations
from cloelib.cosmology import derived_cosmology
from cloelib.observables.clusters.profile import Profile, ProfileNFW, ProfileBMO
from cloelib.observables.clusters.clustering import HaloClustering
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.auxiliary.units import SPEED_OF_LIGHT

# General imports
#import interpax
import numpy as np
from scipy.integrate import simpson as simps
#import jax

"""

## Notes:

- Cluster counts, Cluster profile lensing and Clusters clustering class 

"""

class ClusterStatistics:
    def __init__(
        self, 
        perturbations: Perturbations,
        haloStatistics: HaloStatistics,
        selectionfunction: SelectionFunction,
        profile: Profile,
        clustering: HaloClustering,
        covariance: HaloCovariance,
        z_obs_edges: np.ndarray,
        Lambda_obs_edges: np.ndarray,
        Rad_obs_edges: np.ndarray,
        Lambda_obs_Cxi2_edges: np.ndarray,
        Rad_obs_Cxi2_edges: np.ndarray,
        z_obs_Cxi2_edges: np.ndarray,
        halo_concentration: float,
        k: np.ndarray,
        Mass : np.ndarray,
        Lambda : np.ndarray,
        z: np.ndarray,
        area: float = 10313,
        CG_like_selection: str = 'CC_CWL_Cxi2',
        CG_xi2_cov_selection: str = 'covCC_covCxi2',
#        external_richness_selection_function: str = 'non_CG_ESF',
        bias: str = 'castro23',
        neutrino_cdm: bool = True,        
        ):
        """
        Initializes the cluster counts

        Parameters:
        - ....
        
        """
        self.perturbations = perturbations
        self.background = self.perturbations.background
        self.haloStatistics = haloStatistics
        self.selectionfunction = selectionfunction
        self.profile = profile
        self.clustering = clustering
        self.covariance = covariance 
        
        self.area = area
        self.CG_like_selection = CG_like_selection
        self.CG_xi2_cov_selection = CG_xi2_cov_selection
#        self.CG_xi2_cov_selection = 'non_CG_ESF'
        self.bias = bias
        self.neutrino_cdm = neutrino_cdm
        
        self.z_obs_edges = z_obs_edges
        self.Lambda_obs_edges = Lambda_obs_edges
        self.Rad_obs_edges = Rad_obs_edges
        self.Lambda_obs_Cxi2_edges = Lambda_obs_Cxi2_edges
        self.Rad_obs_Cxi2_edges = Rad_obs_Cxi2_edges
        self.z_obs_Cxi2_edges = z_obs_Cxi2_edges
        
        self.halo_concentration = halo_concentration
        
        # k array (integration variable)              
        self.k = k

        # mass array (integration variable)
        self.Mass = Mass #in Msun h^-1

        # true richness array (integration variable)
        self.Lambda = Lambda

        # true redshift array (integration variable)
        self.z = z
        
        ################### SELECTION FUNCTION ###################

        # selections (???)
        self.l_m_tab_sig_Cxi2 = [31,51]
        self.l_m_tab_sig = [31,31,31,51]
        self.z_tab_sig   = 31  
        
        ################### OBSERVED BINS ###################

        # observed redshift bins for number counts and weak lensing
        #self.z_obs_edges = z_obs_edges #self.theory['obs_specifications']['CG']['z_obs_edges']        
        self.z_obs_div   = len(self.z_obs_edges) - 1         

        # observed richness bins for number counts and weak lensing
        #self.Lambda_obs_edges = Lambda_obs_edges #self.theory['obs_specifications']['CG']['Lambda_obs_edges']
        self.Lambda_obs_div   = len(self.Lambda_obs_edges) - 1

        # observed radial separation bins for weak lensing
        #self.Rad_obs_edges = Rad_obs_edges #self.theory['obs_specifications']['CG']['Rad_obs_edges']
        self.Rad_obs_div   = len(self.Rad_obs_edges) - 1  

        # observed richness bins for clustering
        #self.Lambda_obs_Cxi2_edges = Lambda_obs_Cxi2_edges #self.theory['obs_specifications']['CG']['Lambda_obs_Cxi2_edges']
        self.Lambda_obs_Cxi2_div   = len(self.Lambda_obs_Cxi2_edges)-1  

        # observed radial separation bins for clustering
        #self.Rad_obs_Cxi2_edges = Rad_obs_Cxi2_edges #self.theory['obs_specifications']['CG']['Rad_obs_Cxi2_edges']   
        self.Rad_obs_Cxi2_div   = len(self.Rad_obs_Cxi2_edges) - 1 

        # observed redshift bins for clustering
        #self.z_obs_Cxi2_edges = z_obs_Cxi2_edges #self.theory['obs_specifications']['CG']['z_obs_Cxi2_edges']
        self.z_obs_Cxi2_div   = len(self.z_obs_Cxi2_edges) - 1 
        
        self.alpha_cov_Cxi2 = np.zeros((self.z_obs_Cxi2_div,self.Lambda_obs_Cxi2_div))
        self.beta_cov_Cxi2  = np.ones((self.z_obs_Cxi2_div,self.Lambda_obs_Cxi2_div))
        self.gamma_cov_Cxi2 = np.zeros((self.z_obs_Cxi2_div,self.Lambda_obs_Cxi2_div))  

        #############################################################      

        # volume element at the center of observed redshift bins
        self.dvdzdomega_z1z2 = derived_cosmology.dV_dzdO(self.background, self.z, hubble_units=True)

        # hmf at the center of observed redshift bins
        self.dndm_z = self.haloStatistics.dn_dm(self.z, self.Mass) 
        self.bias_z = self.haloStatistics.bias(self.z, self.Mass)  # only work for virial overdensity        
        
        
    def N_zbin_Lbin_Rbin(self): 
        
        N_zbin_Lbin   = np.zeros((self.z_obs_div, self.Lambda_obs_div))
        cov_zbin_Lbin = np.zeros((self.z_obs_div, self.z_obs_div, self.Lambda_obs_div,self.Lambda_obs_div))
        n_lbdobs_z    = np.zeros(self.Lambda_obs_div,dtype=list)
        Plob_M_z      = np.zeros(self.Lambda_obs_div,dtype=list)
        b_n_lbdobs_z  = np.zeros(self.Lambda_obs_div,dtype=list)
        dV_dzob       = np.zeros((self.z_obs_div, self.Lambda_obs_div),dtype=list)        
        
        if self.CG_like_selection in ['CC','CC_CWL','CC_Cxi2','CC_CWL_Cxi2']:

            # mean observed bins
            z_obs_mid      = 0.5 * (self.z_obs_edges[1:] + self.z_obs_edges[:-1])
            Lambda_obs_mid = 0.5 * (self.Lambda_obs_edges[1:] + self.Lambda_obs_edges[:-1])           


            ##### number counts covariance
            if self.CG_xi2_cov_selection in ['covCC','covCC_covCxi2']:
                
                # spherical harmonic expansion coefficients (covariance)
                L   = 20
                KL  = self.covariance.Kl_coeff()
                #self.rint = np.zeros((self.z_obs_div,len(self.k),L+1))
                
                # power spectrum at the center of observed redshift bins
                pk  = self.haloStatistics.matter_power_spectrum(z_obs_mid, self.k)          
                
                # corrected halo Pk (only 0-th order correction is enough for number counts covariance)
                photoz_corr0 = self.clustering.photoz_rsd_correction(z_obs_mid, 0)[0]  # can neglect richness dependence here
                pk *=  photoz_corr0
                
                # array initialization for covariance
                sab = np.zeros((self.z_obs_div,self.z_obs_div))            
                SN  = np.zeros((self.z_obs_div, self.z_obs_div, self.Lambda_obs_div,self.Lambda_obs_div))
                Nb_zbin_Lbin = np.zeros((self.z_obs_div, self.Lambda_obs_div))

                # richness-independent quantites that can be computed outside the lambda loop
                for z_bin in range(len(self.z_obs_edges)-1):
                    
                    z_tab = np.linspace(self.z_obs_edges[z_bin], self.z_obs_edges[z_bin+1], self.z_tab_sig)
                        
                    sab[z_bin,:(z_bin+1)] = 1/(2*np.pi**2) * simps((self.k**2 * np.sqrt(pk[z_bin]*pk[:(z_bin+1)])) 
                                                                   * self.covariance.cov_window(z_bin, z_tab, KL), self.k, axis=-1)
                    sab[:(z_bin+1),z_bin] = sab[z_bin,:(z_bin+1)]

            for lambda_bin in range(len(self.Lambda_obs_edges)-1):

                l_tab = np.geomspace(self.Lambda_obs_edges[lambda_bin], self.Lambda_obs_edges[lambda_bin+1], self.l_m_tab_sig[lambda_bin])

                # P(lob|ltr,ztr)
                Plob_l_z  = simps(self.selectionfunction.P_lbdobs_lbd(self.z, self.Lambda, l_tab), l_tab, axis=-1) 
#                if external_richness_selection_function == 'CG_ESF':
#                   Plob_l_z  = self.int_Plobltr_Dlob[lambda_bin](self.z, self.Lambda).T  

                # P(ltrM,ztr)
                Pl_M_z  = self.selectionfunction.P_lnlbd(self.z, self.Mass, self.Lambda) 

                # P(lob|M,ztr)
                Plob_M_z[lambda_bin]  = simps(Pl_M_z[:,:,:]*Plob_l_z[:,np.newaxis,:], self.Lambda,axis=-1)  

                # N(lob,ztr)
                n_lbdobs_z[lambda_bin] = simps(Plob_M_z[lambda_bin]*self.dndm_z,self.Mass, axis=1)  

                # N(lob,ztr) * bias(lob,ztr)
                if self.CG_xi2_cov_selection in ['covCC','covCC_covCxi2']: 
                   b_n_lbdobs_z[lambda_bin] = simps(Plob_M_z[lambda_bin] * self.dndm_z * self.bias_z, self.Mass, axis=1) 

                    
                for z_bin in range(len(self.z_obs_edges)-1):
                        
                    z_tab = np.linspace(self.z_obs_edges[z_bin], self.z_obs_edges[z_bin+1], self.z_tab_sig)
    
                    # P(zob|ztr)                  
                    Pzob_z = simps(self.selectionfunction.P_zobs_z(z_tab, self.Lambda_obs_edges[lambda_bin],self.z), z_tab, axis=0)       
    
                    # observed volume element dV/dz_ob
                    dV_dzob[z_bin,lambda_bin] = self.dvdzdomega_z1z2 * Pzob_z * (self.area)*(np.pi**2.0/180.0**2.0) 
    
                    # N(lob,zob)
                    N_zbin_Lbin[z_bin,lambda_bin]  = simps(n_lbdobs_z[lambda_bin] * dV_dzob[z_bin,lambda_bin], self.z, axis=0)   
    
                    if self.CG_xi2_cov_selection in ['covCC','covCC_covCxi2']:
    
                        # N(lob,zob) * bias(lob,zob)
                        Nb_zbin_Lbin[z_bin, lambda_bin] = simps(b_n_lbdobs_z[lambda_bin] * dV_dzob[z_bin,lambda_bin], self.z, axis=0) 
           
                        # shot-noise matrix
                        SN[z_bin,z_bin] = np.diag(N_zbin_Lbin[z_bin]) 
           

            if self.CG_xi2_cov_selection in ['covCC','covCC_covCxi2']:

               # total covariance = shot-noise + sample covariance
               cov_zbin_Lbin = SN + (Nb_zbin_Lbin.reshape(1,self.z_obs_div,1,self.Lambda_obs_div) *
                                     Nb_zbin_Lbin.reshape(self.z_obs_div,1,self.Lambda_obs_div,1) * 
                                     sab.reshape(self.z_obs_div,self.z_obs_div,1,1)) 
                                             
        ##### reduced shear
        g_zbin_Lbin_Rbin = np.zeros(
            ((self.z_obs_div, self.Lambda_obs_div, self.Rad_obs_div)))         
        
        if self.CG_like_selection in ['CC_CWL','CC_CWL_Cxi2'] :
            
            Rad_obs     = self.Rad_obs_edges
            Rad_obs_mid = 0.5 * (Rad_obs[1:] + Rad_obs[:-1])            

            for rad_bin in range(len(Rad_obs_mid)):                        
                        excess_surface_mass_density = self.profile.excess_surface_mass_density(np.atleast_1d(Rad_obs[rad_bin]), self.z, 
                                                                                       self.Mass,self.halo_concentration)
                                                                                                  
                        for lambda_bin in range(len(self.Lambda_obs_edges)-1):
                            excesssurfacemassdensity = simps(Plob_M_z[lambda_bin]*self.dndm_z*np.squeeze(excess_surface_mass_density,axis=2), self.Mass, axis=1)
                                 
                            for z_bin in range(len(self.z_obs_edges)-1):
                                g_zbin_Lbin_Rbin[z_bin,lambda_bin, rad_bin] = (1.0)/N_zbin_Lbin[z_bin,lambda_bin] * simps(
                                    self.profile.m_sig_crit_m1(self.z, z_bin)*dV_dzob[z_bin,lambda_bin]*excesssurfacemassdensity,self.z) 
#                                      
        ### !!!! note that the final number of richness bins is NL=nl+1 ONLY if we have two richness bins,
        ### if nl>2, the effective number of richness bins is NL=factorial(nl)//(factorial(nl-2)*factorial(2)) + nl
        ### this makes the reshape of the matrix more complex. Since we plan to use only two bins, for the moment it is not implemented.

        Cxi2_zbin_Lbin_Rbin = np.zeros((((self.z_obs_Cxi2_div, self.Lambda_obs_Cxi2_div+1, self.Rad_obs_Cxi2_div))))
        cov_Cxi2 = np.zeros((self.z_obs_Cxi2_div, self.z_obs_Cxi2_div, self.Lambda_obs_Cxi2_div + 1, 
                           self.Lambda_obs_Cxi2_div + 1, self.Rad_obs_Cxi2_div, self.Rad_obs_Cxi2_div)) 
                           
        # 2point correlation function
        if self.CG_like_selection in ['CC_Cxi2','CC_CWL_Cxi2']:
            
            # init
            n_lbdobs_z_Cxi2    = np.zeros((self.Lambda_obs_Cxi2_div,len(self.z))) 
            n_b_lbdobs_z_Cxi2  = np.zeros((self.Lambda_obs_Cxi2_div,len(self.z)))
            integ_zbin_lbin  = np.zeros(((self.z_obs_Cxi2_div, self.Lambda_obs_Cxi2_div, len(self.k))))
            
            # clustering bins
            z_obs_Cxi2_mid      = 0.5 * (self.z_obs_Cxi2_edges[1:] + self.z_obs_Cxi2_edges[:-1])
            Lambda_obs_Cxi2_mid = 0.5 * (self.Lambda_obs_Cxi2_edges[1:] + self.Lambda_obs_Cxi2_edges[:-1])
            
            # array initialization for clustering covariance
            V_zob = np.zeros(len(self.z_obs_Cxi2_edges)-1)
            sqrt_Pk_zbin_lbin = np.zeros(((self.z_obs_Cxi2_div, self.Lambda_obs_Cxi2_div, len(self.k))))
            one_over_n_lambdai_lambdaj = np.zeros(((self.z_obs_Cxi2_div, self.Lambda_obs_Cxi2_div, self.Lambda_obs_Cxi2_div)))            
                      
            # spherical shell window function W(R,K) and volume of the shell (compute once outside the redshift loop)
            W_rad, V_rad = self.clustering.WF_ra(z_obs_Cxi2_mid,self.Rad_obs_Cxi2_edges)                 


            # matter power spectrum + IR resummation
            ############
            #### !!!!! ADD IR RESUMMATION (to be implemented? already implemented for galaxy clustering?)
            ############
            pk_IR = self.haloStatistics.matter_power_spectrum(self.z, self.k) 
            
            # LOOP OVER CLUSTERING RICHNESS BINS
            for lambda_bin in range(len(self.Lambda_obs_Cxi2_edges)-1):

                l_tab = np.geomspace(self.Lambda_obs_Cxi2_edges[lambda_bin], 
                                     self.Lambda_obs_Cxi2_edges[lambda_bin+1], self.l_m_tab_sig_Cxi2[lambda_bin])
                
                ## recompute quantities that depend on lambda_obs

                # P(lob|ltr,ztr)
                Plob_l_z_Cxi2 = simps(self.selectionfunction.P_lbdobs_lbd(self.z, self.Lambda, l_tab), l_tab, axis=-1)         

                # P(lob|M,ztr)   
                Plob_M_z_Cxi2 = simps(Pl_M_z[:,:,:] * Plob_l_z_Cxi2[:,np.newaxis,:], self.Lambda,axis=-1)  
                
                # n(lob,ztr) 
                n_lbdobs_z_Cxi2 = simps(Plob_M_z_Cxi2 * self.dndm_z, self.Mass, axis=1)

                # n(lob,ztr) * b(lob,zob) 
                n_b_lbdobs_z_Cxi2 = simps(Plob_M_z_Cxi2 * self.dndm_z * self.bias_z, self.Mass, axis=1)

                # effective halo bias
                b_eff = (n_b_lbdobs_z_Cxi2/n_lbdobs_z_Cxi2)[:,np.newaxis]

                # correct power specrum for photo-z uncertainties and RSD (eqs. 80-83)
                photoz_corr0, photoz_corr1, photoz_corr2 = self.clustering.photoz_rsd_correction(self.z, Lambda_obs_Cxi2_mid[lambda_bin])
                pk_halo = pk_IR * (b_eff**2 * photoz_corr0 + b_eff*photoz_corr1 + photoz_corr2) 
                
                for z_bin in range(len(self.z_obs_Cxi2_edges)-1):
                    
                    z_tab = np.linspace(self.z_obs_Cxi2_edges[z_bin], self.z_obs_Cxi2_edges[z_bin+1], self.z_tab_sig)
                    

                    # P(zob|ztr)                  
                    Pzob_z = simps(self.selectionfunction.P_zobs_z(z_tab, self.Lambda_obs_Cxi2_edges[lambda_bin],self.z), z_tab, axis=0)       

                    # observed volume element dV/dz_ob
                    dV_dzob = self.dvdzdomega_z1z2 * Pzob_z * (self.area)*(np.pi**2.0/180.0**2.0) 

                    # volume of the observed redshift slice
                    V_zob[z_bin] = simps(dV_dzob,self.z,axis=0)  

                    # normalization factor
                    N_int_lbdobs_z_Cxi2 = simps(dV_dzob * n_lbdobs_z_Cxi2, self.z, axis=0) 
                    
                    # power spectrum and shot-noise terms
                    sqrt_Pk_zbin_lbin[z_bin,lambda_bin,:] = simps((dV_dzob*n_lbdobs_z_Cxi2)[:,np.newaxis] * np.sqrt(pk_halo),
                                                                  self.z,axis=0)/N_int_lbdobs_z_Cxi2
                    
                    one_over_n_lambdai_lambdaj[z_bin,lambda_bin,lambda_bin] = V_zob[z_bin]/N_int_lbdobs_z_Cxi2

            
            # cross Pk and shot-noise in two richness bins
            Pk_lambdai_lambdaj = sqrt_Pk_zbin_lbin[:,:,np.newaxis,:]*sqrt_Pk_zbin_lbin[:,np.newaxis,:,:]    # dim = [nz,nl,nl,nk]
            one_over_n_lambdai_lambdaj = one_over_n_lambdai_lambdaj[:,:,:,np.newaxis]                       # dim = [nz,nl,nl,nk]             


            # compute 2point correlation function
            # dim = [nz,nl,nl,nr]
            Cxi2_zbin_Lbin_Rbin_buf = simps((self.k**2./(2.*np.pi**2) * W_rad[:,np.newaxis,np.newaxis,:,:] *
                                           Pk_lambdai_lambdaj[:,:,:,np.newaxis,:]), self.k,axis=-1)
            
            # xi(lambda_i,lambda_j) = xi(lambda_j,lambda_i), so reshape and keep only one of them
            for z_bin in range(self.z_obs_Cxi2_div):           
                for rad_bin in range(self.Rad_obs_Cxi2_div):
                    Cxi2_zbin_Lbin_Rbin[z_bin,:,rad_bin] = Cxi2_zbin_Lbin_Rbin_buf[z_bin,:,:,rad_bin][np.triu_indices(self.Lambda_obs_Cxi2_div)]

#            
            # 2point correlation function covariance
            if self.CG_xi2_cov_selection in ['covCxi2','covCC_covCxi2']:
                ### alpha(z,l), beta(z,l), gamma(z,l) are nuisance parameters to be fitted on (few, ~100) simulations
                ### to correct for bias model inaccuracy, non-poissonian shot-noise and high-order terms
                ### ref values are alpha=0,beta=1,gamma=0
                ### (see Euclid Collaboration: Fumagalli et al. 2022)
                
                # combine and reshape 
                alpha_ij = (1+self.alpha_cov_Cxi2[:,:,np.newaxis,np.newaxis])*(1+self.alpha_cov_Cxi2[:,np.newaxis,:,np.newaxis])
                beta_ij  = self.beta_cov_Cxi2[:,:,np.newaxis,np.newaxis]*self.beta_cov_Cxi2[:,np.newaxis,:,np.newaxis]  

                beta_pk_ij = beta_ij*Pk_lambdai_lambdaj
                alpha_n_ij = alpha_ij*one_over_n_lambdai_lambdaj
                
                
                ### TWO TERMS OF EQ. 73             
                cov_g  = np.zeros((self.z_obs_Cxi2_div,self.Lambda_obs_Cxi2_div,self.Lambda_obs_Cxi2_div,self.Lambda_obs_Cxi2_div,
                                  self.Lambda_obs_Cxi2_div,self.Rad_obs_Cxi2_div,self.Rad_obs_Cxi2_div))
                cov_ng = np.zeros((self.z_obs_Cxi2_div,self.Lambda_obs_Cxi2_div,self.Lambda_obs_Cxi2_div,self.Lambda_obs_Cxi2_div,
                                  self.Lambda_obs_Cxi2_div,self.Rad_obs_Cxi2_div,self.Rad_obs_Cxi2_div))

                for lambda_bin_i in range(self.Lambda_obs_Cxi2_div):
                    for lambda_bin_j in range(self.Lambda_obs_Cxi2_div):

                        for rad_bin in range(self.Rad_obs_Cxi2_div):
                            
                            cov_ng[:,lambda_bin_i,lambda_bin_j,lambda_bin_i,lambda_bin_j,rad_bin,rad_bin] = (simps(self.k**2./(2.*np.pi**2.) 
                            * W_rad[:,rad_bin,:] * beta_pk_ij[:,lambda_bin_i,lambda_bin_j,:], x=self.k) 
                            * (1+self.gamma_cov_Cxi2[:,lambda_bin_i])*one_over_n_lambdai_lambdaj[:,lambda_bin_i,lambda_bin_i,0]
                            * (1+self.gamma_cov_Cxi2[:,lambda_bin_j])*one_over_n_lambdai_lambdaj[:,lambda_bin_j,lambda_bin_j,0]/V_rad[:,rad_bin])

                        for lambda_bin_k in range(self.Lambda_obs_Cxi2_div):
                            for lambda_bin_h in range(self.Lambda_obs_Cxi2_div):

                                # gaussian term
                                cov_g[:,lambda_bin_i,lambda_bin_j,lambda_bin_k,lambda_bin_h,:,:] = simps(self.k**2./(2.*np.pi**2.) 
                                            * W_rad[:,np.newaxis,:,:] * W_rad[:,:,np.newaxis,:]
                                            * (beta_pk_ij+alpha_n_ij)[:,lambda_bin_i,lambda_bin_k,np.newaxis,np.newaxis,:]  
                                            * (beta_pk_ij+alpha_n_ij)[:,lambda_bin_j,lambda_bin_h,np.newaxis,np.newaxis,:], x=self.k, axis=-1)
                               
                                        
              
                ### EQ. 89 + RESHAPE according to 2ptCF
                for z_bin in range(self.z_obs_Cxi2_div):
                    for lambda_bin_i in range(self.Lambda_obs_Cxi2_div):
                        for lambda_bin_j in range(self.Lambda_obs_Cxi2_div):
                            for lambda_bin_k in range(self.Lambda_obs_Cxi2_div):
                                for lambda_bin_h in range(self.Lambda_obs_Cxi2_div):
                                    # NOTE: if more than 2 richness bins, this has to be modified
                                    cov_Cxi2[z_bin,z_bin,lambda_bin_i+lambda_bin_j,lambda_bin_k+lambda_bin_h,:,:] = 1/V_zob[z_bin] * (
                                                            (cov_g+cov_ng)[z_bin,lambda_bin_i,lambda_bin_j,lambda_bin_k,lambda_bin_h,:,:] +
                                                            (cov_g+cov_ng)[z_bin,lambda_bin_i,lambda_bin_j,lambda_bin_h,lambda_bin_k,:,:])

        return N_zbin_Lbin, g_zbin_Lbin_Rbin, Cxi2_zbin_Lbin_Rbin, cov_zbin_Lbin, cov_Cxi2


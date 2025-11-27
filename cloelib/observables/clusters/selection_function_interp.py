## LAST UPDATE 12/09/2025
## as of now, the class is only including functions needed for this specific implementation
# General imports
import jax.numpy as np # type: ignore
import numpy as np # type: ignore
from scipy import integrate, interpolate
from scipy.integrate import simps


class SelectionFunction_interp:

    def __init__(self, A_l: float, B_l: float, C_l: float,
        M_piv: float = 3.0e14, z_piv: float = 0.45,
        lobsNC_edges_NC = np.array([20.,30.,45.,60.,220.]), 
        zobsNC_edges_NC = np.array([0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6]),
        file_selection, Omega_tot)

        r"""
        Class defining the selection function of galaxy clusters, including
        sample purity, completeness, mass-observable relation, and
        uncertainties on observed quantities.

        This is the NEW version, evaluated from the interpolation of SinFonia output file.
        Should be consistent with selection_function.py in the structure.

        Main outputs: 
        - interpolator for tildeI(λtr,ztr,∆λobs,∆zobs) already summed over all tiles and integrated in Delta_Lobs and Delta_zobs
        - mass-observable relation.

        Parameters
        ----------
        A_l : float
            Amplitude of the proxy - mass scaling relation
        B_l : float
            Mass slope of the proxy - mass scaling relation
        C_l : float
            Redshift slope of the proxy - mass scaling relation
        M_piv: float
            Mass pivot in the proxy - mass relation, in h^{-1} Msun
        z_piv: float
            Redshift pivot in the proxy - mass relation
        lobsNC_edges_NC: array
            edges of Lobs bins for Number Counts
        zobsNC_edges_NC: array
            edges of zobs bins for Number Counts
        file_selection: array
            multi-dimensional array storing fits file of the Selection outputted from Sinfonia
            ASSUMPTION: we are reading the fits file outside of this module
        Omega_tot: float
            Total observed area

        """
        self.A_l = A_l
        self.B_l = B_l
        self.C_l = C_l
        self.M_piv = M_piv
        self.z_piv = z_piv
        self.lobsNC_edges_NC = lobsNC_edges_NC
        self.zobsNC_edges_NC = zobsNC_edges_NC 
        self.file_selection = file_selection
        self.Omega_tot = Omega_tot


    def lnlambda(self, z, M):
        r"""
        Mean of the richness-mass relation PDF.

        Computes the theoretical richness at
        the requested true redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.

        Returns
        -------
        lnlambda : numpy.ndarray
            lnlambda[i,j], where i is the true redhshift axis and j the mass axis
        """
        return (
            np.log(self.A_l)
            + self.B_l * np.log(M / (self.M_piv))
            + self.C_l * np.log((1.0 + z[:, np.newaxis]) / (1.0 + self.z_piv))
        )


    def scatter_lnl(self, z, M):
        r"""
        Intrinsic scatter of the proxy - mass relation.

        Computes the scatter of the theoretical richness probability distribution
        at the requested true redshift and mass points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.

        Returns
        -------
        scatter_lnl : numpy.ndarray
            scatter_lnl[i,j], where i is the true redhshift axis and j the mass axis
        """

        return (
            self.sig_A_l
            + self.sig_B_l * np.log(M / (self.M_piv))
            + self.sig_C_l * np.log((1.0 + z[:, np.newaxis]) / (1.0 + self.z_piv))
        )


    def P_lnlbd(self, z, M, Lambda):
        r"""
        Proxy - mass relation PDF.

        Computes the theoretical richness probability distribution
        at the requested true mass, redshift, and richness points.

        Parameters
        ----------
        z: numpy.ndarray
            True redshift points.
        M: numpy.ndarray
            True mass points in h^{-1} Msun.
        Lambda: numpy.ndarray
            True richness points.

        Returns
        -------
        P_lnlbd: numpy.ndarray
            P_lnlbd[i,j,k], where i is the redshift, j is the mass,
            and k is the observed richness index
        """
        lnlambda1 = self.lnlambda(z, M)[:, :, np.newaxis]
        sigmalnl = self.scatter_lnl(z, M)[:, :, np.newaxis]
        Lambda = Lambda[np.newaxis, np.newaxis, :]

        return (
            1.0
            / (Lambda * np.sqrt(2.0 * np.pi * sigmalnl**2.0))
            * np.exp(-((np.log(Lambda) - lnlambda1) ** 2.0) / (2.0 * sigmalnl**2.0))
        )


    ## NEW FUNCTION FROM SINFONIA FILE

    def sel_func_interp(self):


        r"""
        Selection Function from file.
        Computes the integral over Delta_Lobs_NC and Delta_zobs_NC of 
        1/Omega_tot * sum_alpha Omega_alpha*Pα(λobs|λtr,ztr)*Pα(zobs|λtr,ztr)/Pα(λobs,zobs)*Cα(λtr,ztr).
        Builds the interpolators over (ltr,ztr) for all bins in Lobs_NC and zobs_NC.

        Returns
        -------
        integ4d_interp_func: 2d interpolator
        """


        ## Define ranges for Obs arrays: arrays for grid in final NC
        n_lobsNC = len(self.lobsNC_edges)-1
        n_zobsNC = len(self.zobsNC_edges)-1

        ## Find number of tiles from fits file
        num_headers = len(self.file_selection)
        n_tile = int((num_headers-1)/7)

        ## Definition of arrays for Obs and True quantities
        ## ASSUMPTION: all tiles have the same ranges and binning

        delta_zobs_file = self.file_selection[1].header['HIERARCH Z_OBS_STEP']
        delta_lobs_file = self.file_selection[1].header['HIERARCH LAMBDA_OBS_STEP']  
        delta_ztr_file = self.file_selection[1].header['HIERARCH Z_TRUE_STEP']
        delta_ltr_file = self.file_selection[1].header['HIERARCH LAMBDA_TRUE_STEP']

        nsteps_zobs_file = self.file_selection[1].header['NAXIS1']
        nsteps_lobs_file = self.file_selection[1].header['NAXIS2']
        nsteps_ztr_file = self.file_selection[1].header['NAXIS3']
        nsteps_ltr_file = self.file_selection[1].header['NAXIS4']

        zobs_file = np.linspace(self.file_selection[1].header['HIERARCH Z_OBS_START'],self.file_selection[1].header['HIERARCH Z_OBS_END'],
            nsteps_zobs_file,endpoint=True)
        lobs_file = np.linspace(self.file_selection[1].header['HIERARCH LAMBDA_OBS_START'],self.file_selection[1].header['HIERARCH LAMBDA_OBS_END'],
            nsteps_lobs_file,endpoint=True)
        ztr_file = np.linspace(self.file_selection[1].header['HIERARCH Z_TRUE_START'],self.file_selection[1].header['HIERARCH Z_TRUE_END'],
            nsteps_ztr_file,endpoint=True)
        ltr_file = np.linspace(self.file_selection[1].header['HIERARCH LAMBDA_TRUE_START'],self.file_selection[1].header['HIERARCH LAMBDA_TRUE_END'],
            nsteps_ltr_file,endpoint=True)

        ## Check ranges of Obs arrays of the file and compare with NC Obs arrays
        ## Deal with min and max in lobs and zobs:
        ## max(lobs_file) might be < max(lobsNC_edges) so we put an IF condition for now: Prob for lambda > max(lobs_file) = 0
        ## min(lobs_file) might be > min(lobsNC_edges) so we put an IF condition for now: Prob for lambda < min(lobs_file) = 0
        ## max(zobs_file) might be < max(zobsNC_edges) so we put an IF condition for now: Prob for z > max(zobs_file) = 0
        ## min(zobs_file) might be > min(zobsNC_edges) so we put an IF condition for now: Prob for z < min(zobs_file) = 0

        ## max(lobs_file) might be < max(lobsNC_edges)
        if (lobs_file[-1] < self.lobsNC_edges[-1]): 
            n_new = int((self.lobsNC_edges[-1] - lobs_file[-1]) / delta_lobs_file) 
            lobs_added = np.linspace(lobs_file[-1] + delta_lobs_file, self.lobsNC_edges[-1], n_new, endpoint=True)
            lobs_tot_high = np.concatenate((lobs_file,lobs_added),axis=0)
        else:
            lobs_tot_high = lobs_file
        ## min(lobs_file) might be > min(lobsNC_edges)
        if (lobs_file[0] > self.lobsNC_edges[0]):  
            n_new = int((lobs_file[0] - self.lobsNC_edges[0]) / delta_lobs_file) 
            lobs_added = np.linspace(self.lobsNC_edges[0],lobs_file[0]-delta_lobs_file, n_new, endpoint=True)
            lobs_tot = np.concatenate((lobs_added,lobs_tot_high),axis=0)   
            size_add_lmin = n_new
        else:
            lobs_tot = lobs_tot_high
            size_add_lmin = 0
        ## max(zobs_file) might be < max(zobsNC_edges)
        if (zobs_file[-1] < self.zobsNC_edges[-1]):  
            n_new = int((self.zobsNC_edges[-1] - zobs_file[-1]) / delta_zobs_file) 
            zobs_added = np.linspace(zobs_file[-1] + delta_zobs_file, self.zobsNC_edges[-1], n_new, endpoint=True)
            zobs_tot_high = np.concatenate((zobs_file,zobs_added),axis=0)
        else:
            zobs_tot_high = zobs_file
        ## min(zobs_file) might be > min(zobsNC_edges)
        if (zobs_file[0] > self.zobsNC_edges[0]):   
            n_new = int((zobs_file[0] - self.zobsNC_edges[0]) / delta_zobs_file) 
            zobs_added = np.linspace(self.zobsNC_edges[0],zobs_file[0]-delta_zobs_file, n_new, endpoint=True)
            zobs_tot = np.concatenate((zobs_added,zobs_tot_high),axis=0) 
            size_add_zmin = n_new
        else:
            zobs_tot = zobs_tot_high
            size_add_zmin = 0

        nsteps_lobs_tot = len(lobs_tot)
        nsteps_zobs_tot = len(zobs_tot)

        ## Find common index between (lobs_file-->lobs_edges) and (zobs_file-->zobs_edges)

        index_lobsfile_edges = [np.abs(lobs_tot - value).argmin() for value in self.lobsNC_edges]
        index_zobsfile_edges = [np.abs(zobs_tot - value).argmin() for value in self.zobsNC_edges] 


        ## Start loop on the tiles, to: -----------------------------------------------------------
        ## extraxt 4d array, Completeness and Purity for the fits file
        ## evaluate the different ingredients and integrals to obtain tildeI(λtr,ztr,∆λobs,∆zobs)
        ## ASSUMPTION: the input file is normalised

        arrays = {}
        ## Index of P_4d, Compl and Pur for each tile
        index_4d_save = np.zeros(n_tile, dtype=int)
        index_compl_save = np.zeros(n_tile, dtype=int)
        index_pur_save = np.zeros(n_tile, dtype=int)
        ## Array to store area of each tile: Effective area in deg2  
        area_tile = np.zeros(n_tile)

        ## Sum array over all tiles
        sum_a = np.zeros((nsteps_ltr_file,nsteps_ztr_file,nsteps_lobs_tot,nsteps_zobs_tot))

        for it in range(n_tile):

            ## Select index to read 4d array, completeness and purity for the different tiles
            index_4d = 1 + it
            index_compl = 1 + 2*n_tile + it
            index_pur = 1 + 4*n_tile + it
            ## Save index
            index_4d_save[it] = index_4d
            index_compl_save[it] = index_compl
            index_pur_save[it] = index_pur
    
            ## Save 4d array, Completeness and Purity for each tile
            arrays[f"CG_seL_funcT_{it}"] = self.file_selection[index_4d].data
            ## np.shape(arrays[f"CG_seL_funcT_{it}"]): (ltr, ztr, lobs, zobs)
            arrays[f"compl_{it}"] = self.file_selection[index_compl].data
            ## np.shape(arrays[f"compl_{it}"]): (ltr, ztr)
            arrays[f"purity_{it}"] = self.file_selection[index_pur].data
            ## np.shape(arrays[f"purity_{it}"]): (lobs, zobs)

            index_area = 1 + 6*n_tile + it
            area_tile[it] = self.file_selection[index_area].header['HIERARCH EFFECTIVE_AREA']

            ## Re-arrange ranges for 4d array, to match the Obs NC ranges
            arrays[f"CG_seL_funcT_code_{it}"] = np.zeros((nsteps_ltr_file,nsteps_ztr_file,nsteps_lobs_tot,nsteps_zobs_tot))
            arrays[f"CG_seL_funcT_code_{it}"][:,:,size_add_lmin:nsteps_lobs_file+size_add_lmin,size_add_zmin:nsteps_zobs_file+size_add_zmin] = \
            arrays[f"CG_seL_funcT_{it}"]
            ## Re-arrange ranges Purity, to match the Obs NC ranges
            arrays[f"purity_code_{it}"] = np.zeros((nsteps_lobs_tot,nsteps_zobs_tot))
            arrays[f"purity_code_{it}"][size_add_lmin:nsteps_lobs_file+size_add_lmin,size_add_zmin:nsteps_zobs_file+size_add_zmin] = \
            arrays[f"purity_{it}"]

            ## Produce P_alpha(λobs|λtr,ztr) for all tiles
            ## P(lobs|ltr,ztr) = integrate P(lobs,zobs|ltr,ztr) over zobs    
            CG_ricH_seL_funcT_code = integrate.simpson(arrays[f"CG_seL_funcT_code_{it}"], x=zobs_tot, axis=3)
            ## Normalization: we start from P(lobs,zobs | ltr,ztr) that is normalized. Then we integrate on zobs. 
            ## P(lobs|ltr,ztr) in theory is still normalized. But we do not use the full theoretical x range.
            N = integrate.simpson(CG_ricH_seL_funcT_code,x=lobs_tot,axis=2)
            ## To deal with zeros
            norm_CG_ricH_seL_funcT_code = np.zeros_like(CG_ricH_seL_funcT_code, dtype=float)
            norm_CG_ricH_seL_funcT_code = np.divide(CG_ricH_seL_funcT_code, N[:,:,np.newaxis], 
                out=norm_CG_ricH_seL_funcT_code, 
                where = N[:,:,np.newaxis]!=0)
            arrays[f"norm_CG_ricH_seL_funcT_code_{it}"] = norm_CG_ricH_seL_funcT_code
            ## np.shape(arrays[f"norm_CG_ricH_seL_funcT_code_{it}"]): (ltr, ztr, lobs)

            ## Produce P_alpha(zobs|λtr,ztr) for all tiles
            ## P(zobs|ltr,ztr) = integrate P(lobs,zobs|ltr,ztr) over lobs    
            CG_reD_seL_funcT_code = integrate.simpson(arrays[f"CG_seL_funcT_code_{it}"], x=lobs_tot, axis=2) 
            ## Normalization: we start from P(lobs,zobs|ltr,ztr) that is normalized. Then we integrate on lobs. 
            ## P(zobs|ltr,ztr) in theory is still normalized. But we do not use the full theoretical x range.
            N = integrate.simpson(CG_reD_seL_funcT_code,x=zobs_tot,axis=2)
            ## To deal with zeros
            norm_CG_reD_seL_funcT_code = np.zeros_like(CG_reD_seL_funcT_code, dtype=float)
            norm_CG_reD_seL_funcT_code = np.divide(CG_reD_seL_funcT_code, N[:,:,np.newaxis], 
                out=norm_CG_reD_seL_funcT_code, 
                where = N[:,:,np.newaxis]!=0)   
            arrays[f"norm_CG_reD_seL_funcT_code_{it}"] = norm_CG_reD_seL_funcT_code
            ## np.shape(np.shape(arrays[f"norm_CG_reD_seL_funcT_code_{it}"])): (ltr, ztr, zobs)

            ## Evaluate multiplication for each tile
            ## Omega_alpha * Pα(λobs|λtr,ztr) * Pα(zobs|λtr,ztr) / Pα(λobs,zobs) * Cα(λtr,ztr)
            ## Dimensions: (ltr, ztr, lobs)*(ltr, ztr, zobs)*(lobs, zobs)*(ltr, ztr) --> (ltr,ztr,lobs,zobs)
            ## ASSUMPTION: we do not need other rescaling for the effective area Omega_alpha.
            ## Expand dimensions to make shapes align
            a_exp = arrays[f"norm_CG_ricH_seL_funcT_code_{it}"][:, :, :, None]
            b_exp = arrays[f"norm_CG_reD_seL_funcT_code_{it}"][:, :, None, :]
            c_exp = arrays[f"purity_code_{it}"][None, None, :, :]
            d_exp = arrays[f"compl_{it}"][:, :, None, None]  
            ## To avoid dividing by 0: putting elements with 0 values to NaN
            c_exp_safe = np.where(c_exp == 0, np.nan, c_exp)
            arrays[f"mult_{it}"] = area_tile[it] * a_exp*b_exp/c_exp_safe*d_exp
            ## or do we want to put the division to 0? If YES:
            ## arrays[f"mult_{it}"] = np.where(c_exp != 0, a_exp*b_exp*d_exp/c_exp, 0.0)
            ## np.shape(arrays[f"mult_{it}"]) : (ltr,ztr,lobs,zobs)

            ## Produce sum_alpha Omega_alpha*Pα(λobs|λtr,ztr)*Pα(zobs|λtr,ztr)/Pα(λobs,zobs)*Cα(λtr,ztr)
            sum_a += arrays[f"mult_{it}"]
        
        ## Integrate 1/Omega_tot*sum_a = tildeI(λtr,ztr,∆λobs,∆zobs)
        integ4d = np.zeros((n_lobsNC,n_zobsNC,nsteps_ltr_file,nsteps_ztr_file))

        for ltab in range(n_lobsNC):
            l_start = index_lobsfile_edges[ltab]
            l_end = index_lobsfile_edges[ltab + 1]
            lint = lobs_tot[l_start:l_end]
            nl = l_end - l_start    

            for ztab in range(n_zobsNC):
                z_start = index_zobsfile_edges[ztab]
                z_end = index_zobsfile_edges[ztab + 1]
                zint = zobs_tot[z_start:z_end]
                nz = z_end - z_start
        
                integrand = 1/self.Omega_tot * sum_a[:,:,l_start:l_end,z_start:z_end]
                result_z = integrate.simpson(integrand, x=zint, axis=-1)
                result_l = integrate.simpson(result_z, x=lint, axis=-1)        
                integ4d[ltab,ztab,:,:] = result_l

        ## Building Interpolator for tildeI(λtr, ztr, ∆λobs, ∆zobs)
        integ4d_interp_func = [[interpolate.RectBivariateSpline(ltr_file, ztr_file, integ4d[ltab, ztab, :, :])
                               for ztab in range(n_zobsNC)] for ltab in range(n_lobsNC)]

        return integ4d_interp_func


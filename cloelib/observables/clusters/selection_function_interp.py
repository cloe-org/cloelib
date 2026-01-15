## LAST UPDATE 12/09/2025
## as of now, the class is only including functions needed for this specific implementation
# General imports
import jax.numpy as np  # type: ignore
import numpy as np  # type: ignore
from scipy import integrate, interpolate
from scipy.integrate import simps


class SelectionFunction_interp:

    def __init__(
        self,
        A_l: float,
        B_l: float,
        C_l: float,
        M_piv: float = 3.0e14,
        z_piv: float = 0.45,
        sel_cl_data=None,
    ):
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
        lobsNC_edges: array
            edges of Lobs bins for Number Counts
        zobsNC_edges: array
            edges of zobs bins for Number Counts
        sel_cl_data: dict
            Object that read the SEL_CL output file and formats its accordingly. It must contain the keys:

                * z_obs: xxx
                * lambda_obs: xxx
                * z_true: xxx
                * lambda_true: xxx
                * z_obs_step: xxx
                * lambda_obs_step: xxx
                * CG_seL_funcT: xxx
                * completeness: xxx
                * purity: xxx
                * area_tile: xxx
                * Omega_tot: xxx

        """
        self.A_l = A_l
        self.B_l = B_l
        self.C_l = C_l
        self.M_piv = M_piv
        self.z_piv = z_piv
        self._sel_cl_data_original = sel_cl_data

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

    def _get_sel_cl_data_formatted_with_obs_bins(self, lobsNC_edges, zobsNC_edges):
        """Format SEL_CL data with obs bins

        Parameters
        ----------
        lobsNC_edges: array
            edges of Lobs bins for Number Counts
        zobsNC_edges: array
            edges of zobs bins for Number Counts

        Returns
        -------
        sel_cl_data_fmt: dict
            SEL_CL data reshaped with obs bins
        """

        ## Definition of arrays for Obs and True quantities
        ## ASSUMPTION: all tiles have the same ranges and binning

        ## Check ranges of Obs arrays of the file and compare with NC Obs arrays
        ## Deal with min and max in lobs and zobs:
        ## max(lobs_file) might be < max(lobsNC_edges) so we put an IF condition for now: Prob for lambda > max(lobs_file) = 0
        ## min(lobs_file) might be > min(lobsNC_edges) so we put an IF condition for now: Prob for lambda < min(lobs_file) = 0
        ## max(zobs_file) might be < max(zobsNC_edges) so we put an IF condition for now: Prob for z > max(zobs_file) = 0
        ## min(zobs_file) might be > min(zobsNC_edges) so we put an IF condition for now: Prob for z < min(zobs_file) = 0

        # list explicitly all values that will be filled:
        sel_cl_data_fmt = {
            "z_obs": None,
            "lambda_obs": None,
            "z_true": None,
            "lambda_true": None,
            "index_lambda_obs_edges": None,
            "index_z_obs_edges": None,
            "CG_seL_funcT": None,
            "purity": None,
            "completeness": None,
            "I_ltr_ztr_lobs_lobs": None,
            "area_tile": None,
            "Omega_tot": None,
        }

        ################
        # Reshape arrays
        ################

        zobs_fmt, _size_add_zmin = self._redefine_array_with_obs_bins(
            self._sel_cl_data_original["z_obs"],
            self._sel_cl_data_original["z_obs_step"],
            zobsNC_edges,
        )
        lobs_fmt, _size_add_lmin = self._redefine_array_with_obs_bins(
            self._sel_cl_data_original["lambda_obs"],
            self._sel_cl_data_original["lambda_obs_step"],
            lobsNC_edges,
        )

        sel_cl_data_fmt["z_obs"] = zobs_fmt
        sel_cl_data_fmt["lambda_obs"] = lobs_fmt

        ## Find common index between (lobs_file-->lobs_edges) and (zobs_file-->zobs_edges)
        sel_cl_data_fmt["index_lambda_obs_edges"] = [
            np.abs(sel_cl_data_fmt["lambda_obs"] - value).argmin()
            for value in lobsNC_edges
        ]
        sel_cl_data_fmt["index_z_obs_edges"] = [
            np.abs(sel_cl_data_fmt["z_obs"] - value).argmin() for value in zobsNC_edges
        ]

        # Keep true values
        sel_cl_data_fmt["z_true"] = self._sel_cl_data_original["z_true"]
        sel_cl_data_fmt["lambda_true"] = self._sel_cl_data_original["lambda_true"]

        ################
        # Reshape tables
        ################

        # Keep completeness shape
        sel_cl_data_fmt["completeness"] = np.array(
            [comp.copy() for comp in self._sel_cl_data_original["completeness"]]
        )

        # reshape purity and seL_func

        # initialize with zeros
        sel_cl_data_fmt["CG_seL_funcT"] = np.zeros(
            (
                len(self._sel_cl_data_original["CG_seL_funcT"]),
                len(self._sel_cl_data_original["lambda_true"]),
                len(self._sel_cl_data_original["z_true"]),
                len(sel_cl_data_fmt["lambda_obs"]),
                len(sel_cl_data_fmt["z_obs"]),
            )
        )
        sel_cl_data_fmt["purity"] = np.zeros(
            (
                len(self._sel_cl_data_original["CG_seL_funcT"]),
                len(sel_cl_data_fmt["lambda_obs"]),
                len(sel_cl_data_fmt["z_obs"]),
            )
        )

        # find which indices on the table will be filled
        lobs_orig_slice = slice(
            _size_add_lmin,
            _size_add_lmin + len(self._sel_cl_data_original["lambda_obs"]),
        )
        zobs_orig_slice = slice(
            _size_add_zmin, _size_add_zmin + len(self._sel_cl_data_original["z_obs"])
        )

        ## Re-arrange ranges for 4d array, to match the Obs NC ranges
        sel_cl_data_fmt["CG_seL_funcT"][:, :, :, lobs_orig_slice, zobs_orig_slice] = (
            self._sel_cl_data_original[f"CG_seL_funcT"]
        )
        ## Re-arrange ranges Purity, to match the Obs NC ranges
        sel_cl_data_fmt[f"purity"][:, lobs_orig_slice, zobs_orig_slice] = (
            self._sel_cl_data_original[f"purity"]
        )

        ##################
        # Keep area values
        ##################
        sel_cl_data_fmt["area_tile"] = self._sel_cl_data_original["area_tile"].copy()
        sel_cl_data_fmt["Omega_tot"] = self._sel_cl_data_original["Omega_tot"]

        return sel_cl_data_fmt

    def _compute_I_ltr_ztr_lobs_lobs(self, sel_cl_data_fmt):
        """Computes Omega_alpha*Pα(λobs|λtr,ztr)*Pα(zobs|λtr,ztr)/Pα(λobs,zobs)*Cα(λtr,ztr)
        and add it to input dictionary.

        Parameters
        ----------
        sel_cl_data_fmt: dict
            SEL_CL data reshaped with obs bins
        """
        ## Start loop on the tiles, to: -----------------------------------------------------------
        ## extraxt 4d array, Completeness and Purity for the fits file
        ## evaluate the different ingredients and integrals to obtain tildeI(λtr,ztr,∆λobs,∆zobs)
        ## ASSUMPTION: the input file is normalised

        sel_cl_data_fmt["I_ltr_ztr_lobs_lobs"] = np.zeros_like(
            sel_cl_data_fmt["CG_seL_funcT"],
            dtype=float,
        )

        for it in range(len(sel_cl_data_fmt["I_ltr_ztr_lobs_lobs"])):

            ## Select index to read 4d array, completeness and purity for the different tiles

            ## Produce P_alpha(λobs|λtr,ztr) for all tiles
            ## P(lobs|ltr,ztr) = integrate P(lobs,zobs|ltr,ztr) over zobs
            CG_ricH_seL_funcT = integrate.simpson(
                sel_cl_data_fmt["CG_seL_funcT"][it], x=sel_cl_data_fmt["z_obs"], axis=3
            )
            ## Normalization: we start from P(lobs,zobs | ltr,ztr) that is normalized. Then we integrate on zobs.
            ## P(lobs|ltr,ztr) in theory is still normalized. But we do not use the full theoretical x range.
            norm_CG_ricH_seL_funcT = self._normalize_array(
                CG_ricH_seL_funcT,
                normalization=integrate.simpson(
                    CG_ricH_seL_funcT, x=sel_cl_data_fmt["lambda_obs"], axis=2
                ),
            )
            ## np.shape(norm_CG_ricH_seL_funcT): (ltr, ztr, lobs)

            ## Produce P_alpha(zobs|λtr,ztr) for all tiles
            ## P(zobs|ltr,ztr) = integrate P(lobs,zobs|ltr,ztr) over lobs
            CG_reD_seL_funcT = integrate.simpson(
                sel_cl_data_fmt["CG_seL_funcT"][it],
                x=sel_cl_data_fmt["lambda_obs"],
                axis=2,
            )
            ## Normalization: we start from P(lobs,zobs|ltr,ztr) that is normalized. Then we integrate on lobs.
            ## P(zobs|ltr,ztr) in theory is still normalized. But we do not use the full theoretical x range.
            norm_CG_reD_seL_funcT = self._normalize_array(
                CG_reD_seL_funcT,
                normalization=integrate.simpson(
                    CG_reD_seL_funcT, x=sel_cl_data_fmt["z_obs"], axis=2
                ),
            )
            ## np.shape(np.shape(norm_CG_reD_seL_funcT)): (ltr, ztr, zobs)

            ## Evaluate multiplication for each tile
            ## Omega_alpha * Pα(λobs|λtr,ztr) * Pα(zobs|λtr,ztr) / Pα(λobs,zobs) * Cα(λtr,ztr)
            ## Dimensions: (ltr, ztr, lobs)*(ltr, ztr, zobs)*(lobs, zobs)*(ltr, ztr) --> (ltr,ztr,lobs,zobs)
            ## ASSUMPTION: we do not need other rescaling for the effective area Omega_alpha.
            ## Expand dimensions to make shapes align

            ## To avoid dividing by 0: putting elements with 0 values to NaN
            pur_reshaped = sel_cl_data_fmt["purity"][it, None, None, :, :]
            pur_reshaped = np.where(pur_reshaped == 0, np.nan, pur_reshaped)

            sel_cl_data_fmt["I_ltr_ztr_lobs_lobs"][it] = (
                sel_cl_data_fmt["area_tile"][it]
                * norm_CG_ricH_seL_funcT[:, :, :, None]
                * norm_CG_reD_seL_funcT[:, :, None, :]
                / pur_reshaped
                * sel_cl_data_fmt["completeness"][it, :, :, None, None]
            )
            ## or do we want to put the division to 0? If YES:
            ##  sel_cl_data_fmt["I_ltr_ztr_lobs_lobs"][it] = (
            ##    np.where(pur_reshaped != 0, sel_cl_data_fmt["I_ltr_ztr_lobs_lobs"][it], 0.0)
            ##  )

    def sel_func_interp(
        self,
        lobsNC_edges=np.array([20.0, 30.0, 45.0, 60.0, 220.0]),
        zobsNC_edges=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]),
    ):
        r"""
        Selection Function from file.
        Computes the integral over Delta_Lobs_NC and Delta_zobs_NC of
        1/Omega_tot * sum_alpha Omega_alpha*Pα(λobs|λtr,ztr)*Pα(zobs|λtr,ztr)/Pα(λobs,zobs)*Cα(λtr,ztr).
        Builds the interpolators over (ltr,ztr) for all bins in Lobs_NC and zobs_NC.

        Parameters
        ----------
        lobsNC_edges: array
            edges of Lobs bins for Number Counts
        zobsNC_edges: array
            edges of zobs bins for Number Counts

        Returns
        -------
        integ4d_interp_func: 2d interpolator
        """

        # Format sel_cl data with obs bins
        sel_cl_data_fmt = self._get_sel_cl_data_formatted_with_obs_bins(
            lobsNC_edges, zobsNC_edges
        )

        # Compute Omega_alpha*Pα(λobs|λtr,ztr)*Pα(zobs|λtr,ztr)/Pα(λobs,zobs)*Cα(λtr,ztr)
        self._compute_I_ltr_ztr_lobs_lobs(sel_cl_data_fmt)

        sum_a = sel_cl_data_fmt["I_ltr_ztr_lobs_lobs"].sum(axis=0)

        ##############################
        ## tildeI(λtr,ztr,∆λobs,∆zobs)
        ##############################

        ## Define ranges for Obs arrays: arrays for grid in final NC
        n_lobsNC = len(lobsNC_edges) - 1
        n_zobsNC = len(zobsNC_edges) - 1

        ## Integrate 1/Omega_tot*sum_a = tildeI(λtr,ztr,∆λobs,∆zobs)
        ## and build interpolator
        integ4d = np.zeros(
            (
                n_lobsNC,
                n_zobsNC,
                len(sel_cl_data_fmt["lambda_true"]),
                len(sel_cl_data_fmt["z_true"]),
            )
        )
        integ4d_interp_func = []

        for ltab in range(n_lobsNC):
            l_start, l_end = sel_cl_data_fmt["index_lambda_obs_edges"][ltab : ltab + 2]
            lint = sel_cl_data_fmt["lambda_obs"][l_start:l_end]

            integ4d_interp_func.append([])

            for ztab in range(n_zobsNC):
                z_start, z_end = sel_cl_data_fmt["index_z_obs_edges"][ztab : ztab + 2]
                zint = sel_cl_data_fmt["z_obs"][z_start:z_end]

                integrand = (
                    sum_a[:, :, l_start:l_end, z_start:z_end]
                    / sel_cl_data_fmt["Omega_tot"]
                )
                result_z = integrate.simpson(integrand, x=zint, axis=-1)
                result_l = integrate.simpson(result_z, x=lint, axis=-1)
                integ4d[ltab, ztab, :, :] = result_l

                integ4d_interp_func[-1].append(
                    interpolate.RectBivariateSpline(
                        sel_cl_data_fmt["lambda_true"],
                        sel_cl_data_fmt["z_true"],
                        integ4d[ltab, ztab, :, :],
                    )
                )

        return integ4d_interp_func

    def _normalize_array(
        self,
        array,
        normalization,
    ):
        """Safe normalization (does not explode at norm=0).

        Parameters
        ----------
        array : numpy.nparray
            Array to be normalized
        normalization : numpy.nparray
            Normalization values, must have shape = array.shape[:-1]


        Returns
        -------
        norm_array : numpy.ndarray
            Normalized array
        """
        norm_array = np.zeros_like(array, dtype=float)
        norm_array = np.divide(
            array,
            normalization[..., np.newaxis],
            out=norm_array,
            where=normalization[..., np.newaxis] != 0,
        )
        return norm_array

    @staticmethod
    def _redefine_array_with_obs_bins(
        original_array,
        original_array_step,
        obs_bins,
    ):
        ## max(original_array) might be < max(obs_bins)
        if original_array[-1] < obs_bins[-1]:
            n_new = int((obs_bins[-1] - original_array[-1]) / original_array_step)
            array_added = np.linspace(
                original_array[-1] + original_array_step,
                obs_bins[-1],
                n_new,
                endpoint=True,
            )
            array_tot_high = np.concatenate((original_array, array_added), axis=0)
        else:
            array_tot_high = original_array
        ## min(original_array) might be > min(obs_bins)
        if original_array[0] > obs_bins[0]:
            n_new = int((original_array[0] - obs_bins[0]) / original_array_step)
            array_added = np.linspace(
                obs_bins[0],
                original_array[0] - original_array_step,
                n_new,
                endpoint=True,
            )
            array_new = np.concatenate((array_added, array_tot_high), axis=0)
            size_add_min = n_new
        else:
            array_new = array_tot_high
            size_add_min = 0

        return array_new, size_add_min


def read_sel_cl_output(sel_cl_filename, Omega_tot=None):
    """Object to read ouput file from SEL_CL.

    Parameters
    ----------
    file_selection: array
        multi-dimensional array storing fits file of the Selection outputted from Sinfonia
        ASSUMPTION: we are reading the fits file outside of this module
    Omega_tot: float
        Total observed area

    Returns
    -------
    sel_cl_data: dict
        Object that read the SEL_CL output file and formats its accordingly. It must contain the keys:

            * z_obs: xxx
            * lambda_obs: xxx
            * z_true: xxx
            * lambda_true: xxx
            * z_obs_step: xxx
            * lambda_obs_step: xxx
            * CG_seL_funcT: xxx
            * completeness: xxx
            * purity: xxx
            * area_tile: xxx
            * Omega_tot: xxx
    """

    # To be adapted with fitsio - f = fitsio.FITS("your_file.fits")
    file_selection = astropy.io.fits.open(sel_cl_filename)

    # dictionary to store all outputs
    sel_cl_data = {}
    for i, name in enumerate(["Z_OBS", "LAMBDA_OBS", "Z_TRUE", "LAMBDA_TRUE"]):
        sel_cl_data[name.lower()] = np.linspace(
            file_selection[1].header[f"HIERARCH {name}_START"],
            file_selection[1].header[f"HIERARCH {name}_END"],
            file_selection[1].header[f"NAXIS{i+1}"],
            endpoint=True,
        )
    for name in ["Z_OBS", "LAMBDA_OBS"]:
        sel_cl_data[f"{name.lower()}_step"] = file_selection[1].header[
            f"HIERARCH {name}_STEP"
        ]

    # Add selection tables per tile

    ## Start loop on the tiles, to: -----------------------------------------------------------
    ## extraxt 4d array, Completeness and Purity for the fits file

    ## Find number of tiles from fits file
    num_headers = len(file_selection)
    n_tiles = int((num_headers - 1) / 7)

    ## Index of P_4d, Compl and Pur for each tile
    # index_4d_save = np.zeros(n_tiles, dtype=int)
    # index_compl_save = np.zeros(n_tiles, dtype=int)
    # index_pur_save = np.zeros(n_tiles, dtype=int)

    sel_cl_data.update(
        {
            "CG_seL_funcT": [],
            "completeness": [],
            "purity": [],
            "area_tile": np.zeros(n_tiles),
        }
    )

    for it in range(n_tiles):

        ## Select index to read 4d array, completeness and purity for the different tiles
        index_4d = 1 + it
        index_compl = 1 + 2 * n_tiles + it
        index_pur = 1 + 4 * n_tiles + it
        ## Save index
        # index_4d_save[it] = index_4d
        # index_compl_save[it] = index_compl
        # index_pur_save[it] = index_pur

        ## Save 4d array, Completeness and Purity for each tile
        sel_cl_data["CG_seL_funcT"].append(file_selection[index_4d].data)
        ## np.shape(sel_cl_data["CG_seL_funcT"]): (ltr, ztr, lobs, zobs)
        sel_cl_data["completeness"].append(file_selection[index_compl].data)
        ## np.shape(sel_cl_data["completeness"]): (ltr, ztr)
        sel_cl_data["purity"].append(file_selection[index_pur].data)

        index_area = 1 + 6 * n_tiles + it
        sel_cl_data["area_tile"][it] = file_selection[index_area].header[
            "HIERARCH EFFECTIVE_AREA"
        ]

    sel_cl_data["Omega_tot"] = sel_cl_data["area_tile"].sum()

    # convert tables into arrays
    for name in ("CG_seL_funcT", "completeness", "purity"):
        sel_cl_data[name] = np.array(sel_cl_data[name])

    return sel_cl_data


if __name__ == "__main__":

    # lobsNC_edges=np.array([20.0, 30.0, 45.0, 60.0, 220.0]),
    # zobsNC_edges=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]),

    sel_cl_data = {
        "z_obs": np.linspace(0.3, 1.5, 25),
        "lambda_obs": np.linspace(25, 200, 27),
        "z_true": np.linspace(0, 3, 31),
        "lambda_true": np.linspace(5, 300, 29),
    }
    sel_cl_data["z_obs_step"] = (
        sel_cl_data["z_obs"][1:] - sel_cl_data["z_obs"][:-1]
    ).mean()
    sel_cl_data["lambda_obs_step"] = (
        sel_cl_data["lambda_obs"][1:] - sel_cl_data["lambda_obs"][:-1]
    ).mean()

    # tables

    _rich_norm = lambda rich, z, rich_piv: rich / (rich_piv + z)
    _cp_func = lambda rich, z, rich_piv: (
        _rich_norm(rich, z, rich_piv) / (1 + _rich_norm(rich, z, rich_piv))
    )

    sel_cl_data["CG_seL_funcT"] = np.exp(
        -(
            (
                (
                    sel_cl_data["lambda_obs"][None, None, :, None]
                    - sel_cl_data["lambda_true"][:, None, None, None]
                )[None, ...]
                / np.array([10, 10, 10])[:, None, None, None, None]
            )
            ** 2
        )
        - (
            (
                sel_cl_data["z_obs"][None, None, None, :]
                - sel_cl_data["z_true"][None, :, None, None]
            )[None, ...]
        )
        ** 2
    )
    sel_cl_data["completeness"] = _cp_func(
        sel_cl_data["lambda_true"][None, :, None],
        sel_cl_data["z_true"][None, None, :],
        np.array([10, 15, 10])[:, None, None],
    )
    sel_cl_data["purity"] = _cp_func(
        sel_cl_data["lambda_obs"][None, :, None],
        sel_cl_data["z_obs"][None, None, :],
        np.array([10, 10, 15])[:, None, None],
    )

    sel_cl_data["area_tile"] = np.array([8, 9, 10])
    sel_cl_data["Omega_tot"] = sel_cl_data["area_tile"].sum()

    sfi = SelectionFunction_interp(
        A_l=None,
        B_l=None,
        C_l=None,
        sel_cl_data=sel_cl_data,
    )
    sfi.sel_func_interp()

# General imports
import jax.numpy as np  # type: ignore
import numpy as np  # type: ignore
from scipy import integrate, interpolate
from scipy.integrate import simps

from cloelib.observables.clusters.selection_function.lambda_true_distribution import (
    LambdaTrueDistribution,
)


class InterpolatedSelectionFunction:

    def __init__(
        self,
        lambda_true_distribution: LambdaTrueDistribution,
        sel_cl_data=None,
        prob_contains_completeness=True,
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
        lambda_true_distribution: LambdaTrueDistribution,
            Object that contains the distribution of true richness given mass
        sel_cl_data: dict
            Object that read the SEL_CL output file and formats its accordingly. It must contain the keys:

                * area_tile: xxx
                * Omega_tot: xxx
                * arrays:
                    * z_obs: xxx
                    * lambda_obs: xxx
                    * z_true: xxx
                    * lambda_true: xxx
                * tables:
                    * prob_lambda_z_obs: xxx
                    * completeness: xxx
                    * purity: xxx
                * aux:
                    * z_obs_step: xxx
                    * lambda_obs_step: xxx

        prob_contains_completeness : bool
            If sel_cl_data["prob_lambda_z_obs"] already accounts for the completeness.

        """
        self.lambda_true_distribution = lambda_true_distribution
        self._sel_cl_data = sel_cl_data
        self.prob_contains_completeness = prob_contains_completeness

    def _reshape_data_with_obs_bins(self, lambda_obs_edges, z_obs_edges):
        """Reshapes SEL_CL data with obs bins and computes
        Omega_alpha*Pα(λobs|λtr,ztr)*Pα(zobs|λtr,ztr)/Pα(λobs,zobs)*Cα(λtr,ztr)

        Check ranges of Obs arrays of the file and compare with NC Obs arrays.

        Deals with min and max in lobs and zobs:

            * max(lobs_file) might be < max(lambda_obs_edges)
            * min(lobs_file) might be > min(lambda_obs_edges)
            * max(zobs_file) might be < max(z_obs_edges)
            * min(zobs_file) might be > min(z_obs_edges)

        Parameters
        ----------
        lambda_obs_edges: array
            edges of Lobs bins for Number Counts
        z_obs_edges: array
            edges of zobs bins for Number Counts

        Returns
        -------
        out_data: dict
            SEL_CL data reshaped with obs bins

                * area_tile: xxx
                * Omega_tot: xxx
                * arrays:
                    * z_obs: xxx
                    * lambda_obs: xxx
                    * z_true: xxx
                    * lambda_true: xxx
                * tables:
                    * prob_lambda_z_obs: xxx
                    * completeness: xxx
                    * purity: xxx
                    * I_ltr_ztr_lobs_lobs: xxx
                * obs_bins_slices:
                    * z: xxx
                    * lambda: xxx

        Note
        ----
            It assumes all tiles have the same ranges and binning!
        """

        # output instanciated with quantities that remain the same:
        out_data = {
            "area_tile": self._sel_cl_data["area_tile"],
            "Omega_tot": self._sel_cl_data["Omega_tot"],
            "arrays": {
                key: self._sel_cl_data["arrays"][key]
                for key in ("z_true", "lambda_true")
            },
            "tables": {"completeness": self._sel_cl_data["tables"]["completeness"]},
        }

        #################
        # Reshaped arrays
        #################

        out_data["arrays"]["z_obs"] = self._expand_array(
            self._sel_cl_data["arrays"]["z_obs"],
            self._sel_cl_data["aux"]["z_obs_step"],
            z_obs_edges[0],
            z_obs_edges[-1],
        )
        out_data["arrays"]["lambda_obs"] = self._expand_array(
            self._sel_cl_data["arrays"]["lambda_obs"],
            self._sel_cl_data["aux"]["lambda_obs_step"],
            lambda_obs_edges[0],
            lambda_obs_edges[-1],
        )

        #################
        # Reshaped tables
        #################

        # finds which slices correspond to the original arrays
        # i. e. array == expanded_array[slice]
        _lobs_orig_slice = self._get_bin_slices(
            out_data["arrays"]["lambda_obs"],
            self._sel_cl_data["arrays"]["lambda_obs"][[0, -1]],
            endpoint=True,
        )[0]
        _zobs_orig_slice = self._get_bin_slices(
            out_data["arrays"]["z_obs"],
            self._sel_cl_data["arrays"]["z_obs"][[0, -1]],
            endpoint=True,
        )[0]

        # Re-arrange ranges for prob_lambda_z_obs
        out_data["tables"]["prob_lambda_z_obs"] = np.zeros(
            (
                out_data["area_tile"].size,
                out_data["arrays"]["lambda_true"].size,
                out_data["arrays"]["z_true"].size,
                out_data["arrays"]["lambda_obs"].size,
                out_data["arrays"]["z_obs"].size,
            )
        )
        out_data["tables"]["prob_lambda_z_obs"][
            :, :, :, _lobs_orig_slice, _zobs_orig_slice
        ] = self._sel_cl_data["tables"]["prob_lambda_z_obs"]

        # Re-arrange ranges for purity
        out_data["tables"]["purity"] = np.zeros(
            (
                out_data["area_tile"].size,
                out_data["arrays"]["lambda_obs"].size,
                out_data["arrays"]["z_obs"].size,
            )
        )
        out_data["tables"]["purity"][:, _lobs_orig_slice, _zobs_orig_slice] = (
            self._sel_cl_data["tables"]["purity"]
        )

        #################################################################################
        # Compute Omega_alpha*Pα(λobs|λtr,ztr)*Pα(zobs|λtr,ztr)/Pα(λobs,zobs)*Cα(λtr,ztr)
        #################################################################################

        # Evaluate multiplication for each tile
        # Omega_alpha * Pα(λobs|λtr,ztr) * Pα(zobs|λtr,ztr) / Pα(λobs,zobs) * Cα(λtr,ztr)
        # Dimensions: (ltr, ztr, lobs)*(ltr, ztr, zobs)*(lobs, zobs)*(ltr, ztr) --> (ltr,ztr,lobs,zobs)
        # ASSUMPTION: we do not need other rescaling for the effective area Omega_alpha.

        ## To avoid dividing by 0: putting elements with 0 values to NaN
        _pur_reshaped = out_data["tables"]["purity"][:, None, None, :, :]
        _pur_reshaped = np.where(_pur_reshaped == 0, np.nan, _pur_reshaped)

        # compute multiplication
        out_data["tables"]["I_ltr_ztr_lobs_lobs"] = (
            out_data["area_tile"][:, None, None, None, None]
            * out_data["tables"]["prob_lambda_z_obs"]
            / _pur_reshaped
        )
        if not self.prob_contains_completeness:
            out_data["tables"]["I_ltr_ztr_lobs_lobs"] *= out_data["tables"][
                "completeness"
            ][:, :, :, None, None]

        ## Do we want to put the division to 0? If YES:
        ## out_data["tables"]["I_ltr_ztr_lobs_lobs"] = (
        ##   np.where(pur_reshaped != 0, out_data["tables"]["I_ltr_ztr_lobs_lobs"], 0.0)
        ## )

        ##################
        # For integrations
        ##################

        # Find slices that return the correct range for each obs bins
        out_data["obs_bins_slices"] = {
            "lambda": self._get_bin_slices(
                out_data["arrays"]["lambda_obs"], lambda_obs_edges
            ),
            "z": self._get_bin_slices(out_data["arrays"]["z_obs"], z_obs_edges),
        }

        return out_data

    def _build_windows_interpolators(self, lambda_obs_edges, z_obs_edges):
        r"""
        Selection Function from file.
        Computes the integral over Delta_Lobs_NC and Delta_zobs_NC of
        1/Omega_tot * sum_alpha Omega_alpha*Pα(λobs|λtr,ztr)*Pα(zobs|λtr,ztr)/Pα(λobs,zobs)*Cα(λtr,ztr).
        Builds the interpolators over (ltr,ztr) for all bins in Lobs_NC and zobs_NC.


        ..math:
            W_{\Delta\lambda_{\rm obs}, \Delta z_{\rm obs}}(\lambda_{\rm true}, z_{\rm true}) =
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            \int_{\Delta z_{\rm obs}}d z_{\rm obs}
            P(\lambda_{\rm obs}, z_{\rm obs}|\lambda_{\rm true}, z_{\rm true})
            \frac{c(\lambda_{\rm true}, z_{\rm true})}{p(\rm obs}, z_{\rm obs})}


        Parameters
        ----------
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.

        Returns
        -------
        integ4d_interp_func: 2d interpolator
        """

        # Format sel_cl data with obs bins

        sel_cl_data_fmt = self._reshape_data_with_obs_bins(
            lambda_obs_edges, z_obs_edges
        )

        # Integrate sum_a/Omega_tot = tildeI(λtr,ztr,∆λobs,∆zobs)

        tilde_I = (
            sel_cl_data_fmt["tables"]["I_ltr_ztr_lobs_lobs"].sum(axis=0)
            / sel_cl_data_fmt["Omega_tot"]
        )

        integ4d = np.zeros(
            (
                len(lambda_obs_edges) - 1,
                len(z_obs_edges) - 1,
                len(sel_cl_data_fmt["arrays"]["lambda_true"]),
                len(sel_cl_data_fmt["arrays"]["z_true"]),
            )
        )
        for ltab, l_slice in enumerate(sel_cl_data_fmt["obs_bins_slices"]["lambda"]):
            for ztab, z_slice in enumerate(sel_cl_data_fmt["obs_bins_slices"]["z"]):
                integ4d[ltab, ztab, :, :] = integrate.simpson(
                    integrate.simpson(
                        tilde_I[:, :, l_slice, z_slice],
                        x=sel_cl_data_fmt["arrays"]["z_obs"][z_slice],
                        axis=-1,
                    ),
                    x=sel_cl_data_fmt["arrays"]["lambda_obs"][l_slice],
                    axis=-1,
                )

        # build interpolator

        integ4d_interp_func = [
            [
                interpolate.RectBivariateSpline(
                    sel_cl_data_fmt["arrays"]["lambda_true"],
                    sel_cl_data_fmt["arrays"]["z_true"],
                    integ4d_lobs_zobs,
                )
                for integ4d_lobs_zobs in integ4d_lobs
            ]
            for integ4d_lobs in integ4d
        ]

        return integ4d_interp_func

    def window_redshift_richness_observed(
        self, z_obs_edges, lambda_obs_edges, z_true, lambda_true
    ):
        r"""
        Computes the window function for observed redshift and richness bins, i. e.:


        ..math:
            W_{\Delta\lambda_{\rm obs}, \Delta z_{\rm obs}}(M, z_{\rm true}) =
            \int_{0}^{\infty}d\lambda_{\rm true}
            P(\lambda_{\rm true}|M, z_{\rm true})
            \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs}
            \int_{\Delta z_{\rm obs}}d z_{\rm obs}
            P(\lambda_{\rm obs}, z_{\rm obs}|\lambda_{\rm true}, z_{\rm true})
            \frac{c(\lambda_{\rm true}, z_{\rm true})}{p(\rm obs}, z_{\rm obs})}



        Computes the integral over Delta_Lobs_NC and Delta_zobs_NC of
        1/Omega_tot * sum_alpha Omega_alpha*Pα(λobs|λtr,ztr)*Pα(zobs|λtr,ztr)/Pα(λobs,zobs)*Cα(λtr,ztr).
        Builds the interpolators over (ltr,ztr) for all bins in Lobs_NC and zobs_NC.

        Parameters
        ----------
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_true : numpy.ndarray
            True richness to compute the window.
        z_true : numpy.ndarray
            True redshift to compute the window.

        Returns
        -------
        numpy.ndarray
            Window function for observed redshift and richness bins.
            Dimensions: (z_obs_edges, lambda_obs_edges, z_true, lambda_true)
        """
        ################################################
        # Compute P(Delta lobs, Delta zobs|ltrue, ztrue)
        ################################################

        interpolators = self._build_windows_interpolators(lambda_obs_edges, z_obs_edges)

        window_lambda_true = np.zeros(
            len(lambda_obs_edges) - 1,
            len(z_obs_edges) - 1,
            lambda_true.size,
            z_true.size,
        )
        for ltab, interp_lobs in enumerate(interpolators):
            for ztab, interp_lobs_zobs in enumerate(interp_lobs):
                window_lambda_true[ltab, ztab] = interp_lobs_zobs(lambda_true, z_true)

        # chage order of axes to (z_obs_edges, lambda_obs_edges, z_true, lambda_true)
        window_lambda_true = window_lambda_true.transpose(1, 0, 3, 2)

        ################################################
        # Compute P(Delta lobs, Delta zobs|mass, ztrue)
        ################################################

        pdf_mass_richness_scaling = self.lambda_true_distribution.prob_richness(
            z_true, mass, lambda_true
        )  # (z, M, lambda_true)

        return simps(
            pdf_mass_richness_scaling[np.newaxis, np.newaxis, :, :, :]
            * window_lambda_true[:, :, :, np.newaxis, :],
            x=lambda_true,
            axis=-1,
        )

    #######
    # Utils
    #######

    @staticmethod
    def _expand_array(array, array_step, lower_value, upper_value):
        """
        Expands lower/upper boundaries of array.

        Parameters
        ----------
        array : np.ndarray
            Original array
        array_step : float
            Step size of original array
        lower_value : float
            Lower value for expansion. Not used if array.min()<lower_value.
        upper_value : float
            Upper value for expansion. Not used if array.max()>upper_value.

        Returns
        -------
        np.ndarray
            Array with expanded boundaries
        """
        upper_addition = np.arange(
            array[-1] + array_step, upper_value + array_step, array_step
        )
        # tick here: make a decreasing array and flip it
        lower_addition = np.arange(
            array[0] - array_step, lower_value - array_step, -array_step
        )[::-1]

        return np.append(lower_addition, np.append(array, upper_addition))

    @staticmethod
    def _get_bin_slices(array, bins_edges, endpoint=False):
        """
        Finds slices that return the correct range for each bin.

        Parameters
        ----------
        array : np.ndarray
            Original array
        bins_edges : list
            Edges of bins.
        endpoint : bool
            Include upper value of edges in slices.

        Returns
        -------
        list[slice]
            List of slices that returns the values in the array
            for each bin of bins_edges.
        """
        # first, find indices at the bin edges
        ind_edges = np.abs(
            array[np.newaxis, :] - np.array(bins_edges)[:, np.newaxis]
        ).argmin(axis=1)

        shift = 1 if endpoint else 0

        return [slice(low, high + shift) for low, high in zip(ind_edges, ind_edges[1:])]


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
        Object that read the SEL_CL output file and formats its accordingly. It will contain the keys:

                * area_tile: xxx
                * Omega_tot: xxx
                * arrays:
                    * z_obs: xxx
                    * lambda_obs: xxx
                    * z_true: xxx
                    * lambda_true: xxx
                * tables:
                    * prob_lambda_z_obs: xxx
                    * completeness: xxx
                    * purity: xxx
                * aux:
                    * z_obs_step: xxx
                    * lambda_obs_step: xxx
    """

    # To be adapted with fitsio - f = fitsio.FITS("your_file.fits")
    from astropy.io import fits

    file_selection = fits.open(sel_cl_filename)

    # dictionary to store all outputs
    sel_cl_data = {}

    ############
    # get arrays
    ############

    sel_cl_data["arrays"] = {
        name.lower(): np.linspace(
            file_selection[1].header[f"HIERARCH {name}_START"],
            file_selection[1].header[f"HIERARCH {name}_END"],
            file_selection[1].header[f"NAXIS{i+1}"],
            endpoint=True,
        )
        for i, name in enumerate(["Z_OBS", "LAMBDA_OBS", "Z_TRUE", "LAMBDA_TRUE"])
    }

    ############
    # get tables
    ############

    def get_hdu(extname):
        for hdu in file_selection:
            if hdu.header.get("EXTNAME") == extname:
                return hdu
        raise ValueError(f"Missing EXTNAME={extname} hdu from SEL_CL file!")

    # Find number of tiles from fits file
    # this number will eventually be at file_selection[0].header
    n_tiles = int((len(file_selection) - 1) / 7)

    sel_cl_data["tables"] = {
        name: np.array([get_hdu(f"{extname_pref}{it}").data for it in range(n_tiles)])
        for name, extname_pref in (
            ("prob_lambda_z_obs", "PROB_LAMBDA_Z_OBS_TRUE_"),  # (ltr, ztr, lobs, zobs)
            ("completeness", "COMP_LAMBDA_Z_TRUE_TRUE_"),  # (ltr, ztr)
            ("purity", "PURITY_LAMBDA_Z_OBS_OBS_"),  # (lobs, zobs)
        )
    }

    ##########
    # aux data
    ##########

    # step size of obs quantities
    sel_cl_data["aux"] = {
        f"{name.lower()}_step": file_selection[1].header[f"HIERARCH {name}_STEP"]
        for name in ["Z_OBS", "LAMBDA_OBS"]
    }

    # Tile areas
    sel_cl_data["area_tile"] = np.array(
        [
            get_hdu(f"AREA_{it}").header["HIERARCH EFFECTIVE_AREA"]
            for it in range(n_tiles)
        ]
    )

    if Omega_tot is None:
        Omega_tot = sel_cl_data["area_tile"].sum()
    sel_cl_data["Omega_tot"] = Omega_tot

    return sel_cl_data


if __name__ == "__main__":

    # Read data
    import sys
    from cloelib.observables.clusters.selection_function.lambda_true_distribution import (
        LognormalPowerLawLambdaTrueDistribution,
    )

    print("Test with SEL_CL data")
    if len(sys.argv) == 1:
        raise ValueError("Missing SEL_CL input file")
    in_file = sys.argv[1]

    sfi = InterpolatedSelectionFunction(
        lambda_true_distribution=LognormalPowerLawLambdaTrueDistribution(
            A_l=None,
            B_l=None,
            C_l=None,
            sig_A_l=None,
            sig_B_l=None,
            sig_C_l=None,
        ),
        sel_cl_data=read_sel_cl_output(in_file, Omega_tot=None),
    )
    interps = sfi._build_windows_interpolators(
        lambda_obs_edges=np.array([20.0, 30.0, 45.0, 60.0, 220.0]),
        z_obs_edges=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]),
    )
    print(interps[1][1]([10, 20, 30, 40], [0.3, 0.31]))

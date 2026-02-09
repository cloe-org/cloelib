# General imports
import numpy as np
from scipy.integrate import simpson as simps

# cloelib imports
from cloelib.cosmology import derived_cosmology
from cloelib.cosmology.cosmology import Perturbations
from cloelib.observables.clusters.covariance import HaloCovariance
from cloelib.observables.clusters.halo_abundance import HaloAbundance

# import jax

"""

## Notes :

- Cluster statistics modeling

"""


class ClusterStatisticsModeling:
    """Object to compute cluster statistics modeling

    Attributes
    ----------
    tabulated_integrands : dict
        Dictionary with tables that will be used for integrations. Contains :

            * k (numpy.ndarray) : Values of k to be used in integrations
            * M (numpy.ndarray) : Values of mass to be used in integrations
            * lambda_true (numpy.ndarray) : Values of true richness to be used in integrations
            * ztrue (numpy.ndarray) : Values of true redshift to be used in integrations
            * PDF_mass_richness_scaling (numpy.ndarray) : Values for P(lambda_true|M, ztrue)
            * dv/dz(ztrue) (numpy.ndarray) : Values for volume element at each redshift
            * dn/dM(ztrue,M) (numpy.ndarray) : Values for the halo mass function dn/dmdz
            * bias(ztrue,M) (numpy.ndarray) : Values for the halo bias halo_bias
            * dk (numpy.ndarray) : Kernel k^2/2*pi^2 to be used in k integrations
    """

    def __init__(
        self,
        halo_abundance: HaloAbundance,
        integ_k_arr: np.ndarray,
        integ_mass_arr: np.ndarray,
        integ_lambda_true_arr: np.ndarray,
        integ_ztrue_arr: np.ndarray,
        area: float = 10313,
    ):
        """
        Initialize the cluster statistics

        Parameters
        ----------
        HaloAbundance : HaloAbundance
            Halo mass function and bias object
        integ_k_arr : numpy.ndarray
            Values of k to be used in integrations, stored in tabulated_integrands
        integ_mass_arr : numpy.ndarray
            Values of mass to be used in integrations, stored in tabulated_integrands
        integ_lambda_true_arr : numpy.ndarray
            Values of true richness to be used in integrations, stored in tabulated_integrands
        integ_ztrue_arr : numpy.ndarray
            Values of true redshift to be used in integrations, stored in tabulated_integrands
        area : float
            Effective area of the survey in deg2.
        """
        # observable objects
        self.halo_abundance = halo_abundance

        # check if the integration points lie within the interpolation ranges
        if self.matter_statistics.interpolate_pk:
            z_knots, k_knots = self.matter_statistics.Pk_interp.get_knots()
            if (
                integ_ztrue_arr.min() <= z_knots.min()
                or integ_ztrue_arr.max() >= z_knots.max()
            ):
                raise ValueError(
                    "integ_ztrue_arr points lie outside the P(k,z) interpolation range."
                )
            if integ_k_arr.min() <= k_knots.min() or integ_k_arr.max() >= k_knots.max():
                raise ValueError(
                    "integ_k_arr points lie outside the P(k,z) interpolation range."
                )

        if self.matter_statistics.interpolate_da:
            z_knots = self.matter_statistics.da_interp.get_knots()
            if (
                integ_ztrue_arr.min() <= z_knots.min()
                or integ_ztrue_arr.max() >= z_knots.max()
            ):
                raise ValueError(
                    "integ_ztrue_arr points lie outside the D_A interpolation range."
                )

        # integration tables
        self.tabulated_integrands = {
            "k": integ_k_arr,  # k array
            "M": integ_mass_arr,  # mass array in Msun h^-1
            "lambda_true": integ_lambda_true_arr,  # true richness array
            "ztrue": integ_ztrue_arr,  # true redshift array
            # volume element at each point of z array
            "dv/dz(ztrue)": derived_cosmology.dV_dzdO(
                self.matter_statistics.perturbations.background,
                integ_ztrue_arr,
                hubble_units=True,
            )
            * area
            * (np.pi**2.0 / 180.0**2.0),
            # hmf at the center of observed redshift bins
            "dn/dM(ztrue,M)": self.halo_abundance.dn_dm(
                integ_ztrue_arr, integ_mass_arr
            ),
            # halo bias at the center of observed redshift bins
            # only work for virial overdensity
            "bias(ztrue,M)": self.halo_abundance.bias(integ_ztrue_arr, integ_mass_arr),
            # kernel for integration in k
            "dk": integ_k_arr**2.0 / (2.0 * np.pi**2),
        }

    @property
    def matter_statistics(self):
        return self.halo_abundance.core.matter_statistics

    # ----------------------------
    # cluster statistics functions
    # ----------------------------

    def window_z_observed(self, selection_function, z_obs_edges, lambda_obs_edges):
        r"""Compute the window function of each observed redshift bin, given by:

        ..math:
            W_{\Delta z_{\rm obs}}(\lambda_{\rm obs}, z_{\rm true}) = \int_{\Delta z_{\rm obs}}dz_{\rm obs} P(z_{\rm obs}|\lambda_{\rm obs}, z_{\rm true})

        Parameters
        ---------
        selection_function : SelectionFunction
            Selection function object
        z_obs_edges : numpy.ndarray
            Edges of redshift bins for the integration.
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.

        Returns
        -------
        window_z_obs : numpy.ndarray
            Integral of P(z_obs|lambda_obs, ztrue) in z_obs bins,
            where (ztrue) are the values in self.tabulated_integrands.
            Dimensions: (z_obs_edges, lambda_obs_edges, ztrue).
        """
        return selection_function.window_z_observed(
            z_obs_edges,
            lambda_obs_edges,
            self.tabulated_integrands["ztrue"],
            self.tabulated_integrands["lambda_true"],
        )

    def window_richness_observed(self, selection_function, lambda_obs_edges):
        r"""Compute the window function of each observed richness bin, given by:

        ..math:
            W_{\Delta\lambda_{\rm obs}}(M, z_{\rm true}) = \int_{\Delta\lambda_{\rm obs}}d\lambda_{\rm obs} P(\lambda_{\rm obs}|M, z_{\rm true})

        Parameters
        ----------
        selection_function : SelectionFunction
            Selection function object
        lambda_obs_edges : numpy.ndarray
            Edges of richness bins for the integration.

        Returns
        -------
        window_lambda_obs : numpy.ndarray
            Integral of P(lambda_obs|M, ztrue) in lambda_obs bins,
            where (M, ztrue) are the values in self.tabulated_integrands.
            Dimensions: (lambda_obs_edges, ztrue, M).
        """

        return selection_function.window_richness_observed(
            lambda_obs_edges,
            self.tabulated_integrands["ztrue"],
            self.tabulated_integrands["M"],
            self.tabulated_integrands["lambda_true"],
        )

    # ---------------------
    # integration functions
    # ---------------------

    def integrate_probe_function_in_k(self, kernel):
        """Integrate the kernel in k.

        Parameters
        ----------
        kernel : numpy.ndarray
            Quantity to be integrated in k space. Can be multidimensional, but
            the last dimension must be of size (k) from self.tabulated_integrands.

        Returns
        -------
        integrated_kernel : numpy.ndarray
            Quantity integrated in k, dimension same as input
            minus the last one.


        Note
        ----
            This exist as a function on its own for the cluster clustering covariange
            computation, somehow it is faster than using integrate_probe_function_in_k_space,
            to be investigated.
        """
        return simps(kernel, x=self.tabulated_integrands["k"])

    def integrate_probe_function_in_dk(self, kernel):
        """Integrate the kernel in k space with a k^2/2pi kernel.

        Parameters
        ----------
        kernel : numpy.ndarray
            Quantity to be integrated in k space. Can be multidimensional, but
            the last dimension must be of size (k) from self.tabulated_integrands.

        Returns
        -------
        integrated_kernel : numpy.ndarray
            Quantity integrated in k space, dimension same as input
            minus the last one.
        """
        return self.integrate_probe_function_in_k(
            kernel * self.tabulated_integrands["dk"]
        )

    def integrate_probe_function_in_mass(self, probe_function, window_lambda_obs):
        """Integrate over mass convolving with the halo mass function.

        Parameters
        ----------
        probe_function : numpy.ndarray
            Kernel to be integrated in mass and convoluted with observed richness bins.
            Must be dimension (ztrue, M, ...) with (ztrue, M) from self.tabulated_integrands.
        window_lambda_obs : numpy.ndarray
            Integral of P(lambda_obs|M, ztrue) in lambda_obs bins,
            where (M, ztrue) are the values in self.tabulated_integrands.
            Dimensions: (lambda_obs, ztrue, M)

        Returns
        -------
        integrated_probe_function : numpy.ndarray
            Quantity integrated in mass with the halo mass function and convoluted
            with observed richness bins. Dimension (lambda_obs, ztrue, ...),
            with (ztrue) from self.tabulated_integrands.
        """
        # Add lambda_obs_edges dimension to probe_function
        _probe_function = probe_function[np.newaxis, ...]

        # to make window_lambda_obs, hmf same shape as probe_function
        extra_axes = tuple(range(3, 3 + len(_probe_function.shape[3:])))

        # reshape window_lambda_obs
        _window_lambda_obs = np.expand_dims(window_lambda_obs, axis=extra_axes)

        # reshape HMF
        _hmf = np.expand_dims(
            self.tabulated_integrands["dn/dM(ztrue,M)"], axis=(0, *extra_axes)
        )

        # integral of P(lambda_obs|M, z)*dn/dM on lambda_obs bins and mass : (lambda_obs_edges, z)
        integrated_probe_function = simps(
            _probe_function * _hmf * _window_lambda_obs,
            x=self.tabulated_integrands["M"],
            axis=2,
        )
        return integrated_probe_function

    def integrate_probe_function_in_redshift(self, probe_function, window_z_obs):
        """Integrate in true volume dv/dz(ztrue) a probe_function that has been
        binned in observed richness, in each observed redsfhit bin.

        Parameters
        ----------
        probe_function : numpy.ndarray
            Kernel binned in observed richness to be integrated in true redshift, and
            convoluted with observed redshift bins. Must be dimension
            (lambda_obs, ztrue, ...) with (ztrue) from self.tabulated_integrands.
        window_z_obs : numpy.ndarray
            Integral of P(z_obs|lambda_obs, ztrue) in z_obs bins,
            where (ztrue) are the values in self.tabulated_integrands.
            Dimensions: (z_obs, lambda_obs, ztrue, ...) with (ztrue) in tabulated_integrands.

        Returns
        -------
        integrated_probe_function : numpy.ndarray
            Quantity integrated in true redshift for each observed bin.
            Dimension (z_obs, lambda_obs, ...).
        """
        # Add z_obs_edges dimension to probe_function
        _probe_function = probe_function[np.newaxis, ...]

        # to make window_z_obs, dvdz same shape as probe_function
        extra_axes = tuple(range(3, 3 + len(_probe_function.shape[3:])))

        # reshape window_z_obs
        _window_z_obs = np.expand_dims(window_z_obs, axis=extra_axes)

        # reshape dvdz
        _dvdz = np.expand_dims(
            self.tabulated_integrands["dv/dz(ztrue)"], axis=(0, 1, *extra_axes)
        )

        # output : (z_obs, lambda_obs_edges)
        integrated_probe_function = simps(
            _probe_function * _dvdz * _window_z_obs,
            x=self.tabulated_integrands["ztrue"],
            axis=2,
        )
        return integrated_probe_function

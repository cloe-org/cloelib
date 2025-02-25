import numpy as np


class Masking:

    def __init__(self, data: dict, obs_spec: dict):
        r"""Class constructor
        Parameters
        ----------
        data: dict
            Data dictionary
        obs_spec: dict
            Specification dictionary
        """
        self.data = data
        self.obs_spec = obs_spec
        self.use_GCspectro = obs_spec['selection']['GCspectro']['GCspectro']

        self.data_vector_GCspectro = None
        self.covariance_GCspectro = None
        self.masking_vector_GCspectro = None

        if self.use_GCspectro:
            self._create_data_vector_GCspectro()
            self._create_covariance_GCspectro()
            self._create_masking_vector_GCspectro()

    def _create_data_vector_GCspectro(self):
        r"""Arranges the GCspectro data into a single data vector
        """
        datavec = []
        for z in self.data['GCspectro'].keys():
            for ell in [0,2,4]:
                datavec = np.append(datavec, self.data['GCspectro'][z]
                                    [f'pk{ell}'])
        self.data_vector_GCspectro = datavec

    def _create_covariance_GCspectro(self):
        r"""Arranges the GCspectro covariances into a single covariance matrix
        """
        covnumsc = []
        for z in self.data['GCspectro'].keys():
            covnumsc.append(3 * len(self.data['GCspectro'][z]['k']))

        covfull = np.zeros([sum(covnumsc), sum(covnumsc)])

        ind = 0
        for i,z in enumerate(self.data['GCspectro'].keys()):
            covfull[ind:ind+covnumsc[i], ind:ind+covnumsc[i]] = \
                self.data['GCspectro'][z]['cov']
            ind += covnumsc[i]

        self.covariance_GCspectro = covfull

    def _create_masking_vector_GCspectro(self):
        r"""Computes the masking vector for GCspectro
        (could be merged with create_data_vector_GCspectro)
        """
        GCspectro_vec = np.array([], dtype=bool)
        redshifts = self.data['GCspectro'].keys()

        for i,z in enumerate(redshifts):
            k = self.data['GCspectro'][z]['k']
            for ell in [0,2,4]:
                accepted_k = np.array(
                    self.obs_spec['scale_cuts']['GCspectro'][f'bin{i+1}']
                    [f'ell{ell}'])
                GCspectro_vec = np.concatenate(
                    (GCspectro_vec, self._get_masking(k, accepted_k)),
                    axis=None)

        self.masking_vector_GCspectro = GCspectro_vec

    def _get_masking(self, arr: np.ndarray, interval: list):
        r""" Get a 1/0 mask for the elements of arr contained in interval
        Parameters
        ----------
        arr: numpy.ndarray
            Input array
        interval: list
            Edges defining the masking region
        """
        return ((arr >= interval[0]) & (arr <= interval[1]))

    def get_masked_data_vector_GCspectro(self):

        return self.data_vector_GCspectro[self.masking_vector_GCspectro]

    def get_masked_covariance_GCspectro(self):

        return (self.covariance_GCspectro[self.masking_vector_GCspectro]
                [:,self.masking_vector_GCspectro])

    def get_masked_theory_vector_GCspectro(self, arr):

        return arr[self.masking_vector_GCspectro]

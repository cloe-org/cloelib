# -*- coding: utf-8 -*-
"""FFTLOG MODULE

Class to perform integrals with the FFTLog algorithm.
Class to implement the  Fast Fourier Transform in logarithmic space (FFTLog)
algorithm to perform the Fourier transform of a periodic sequence of
logarithmically spaced points. This is based on the work published in
`Hamilton (2000) <https://arxiv.org/abs/astro-ph/9905191v4>` and
`Fang et al. (2020) <https://arxiv.org/abs/1911.11947>`
"""

import numpy as np
from numpy.fft import rfft, irfft
from scipy.special import gamma


def _log_extrap(x, N_extrap_begin, N_extrap_end):
    r"""Log extrapolation

    This function takes as input an array :math:`x`, evaluates its
    behaviour at extrema and use them to logarithmically extrapolate
    the array. The :math:`i`-th point is generated at the beginning/end of
    the array according to
    :math:`x_{i}=x_{0} \exp \left(i \Delta_{\ln x}\right)`, where :math:`x_0`
    is the first/last value and :math:`\Delta_{\ln x}` is the logarithm of the
    ratio of the first/last two values.

    Parameters
    ----------
    x: numpy.ndarray
        Array to be extrapolated

    N_extrap_begin: int
        Number of points to add at the beginning of the input array

    N_extrap_end: int
        Number of points to add at the end of the input array

    Returns
    -------
    x_extrap: array
        Returns the extrapolated array
    """

    low_x = high_x = []
    if (N_extrap_begin):
        dlnx_low = np.log(x[1] / x[0])
        low_x = x[0] * np.exp(dlnx_low * np.arange(-N_extrap_end, 0))
    if (N_extrap_end):
        dlnx_high = np.log(x[-1] / x[-2])
        high_x = x[-1] * np.exp(dlnx_high * np.arange(1, N_extrap_end + 1))

    x_extrap = np.hstack((low_x, x, high_x))
    return x_extrap


def _c_window(n, n_cut):
    r"""_c_window

    One-side window function of c_m,
    Adapted from Eq.(C1) in
    `McEwen et al. (2016) <https://arxiv.org/abs/1603.04826>`_.

    .. math::
        W(x)= \begin{cases}
        \frac{x_{\max }-x}{x_{\max }-x_{\text {cut }}}-\frac{1}{2 \pi}
        \sin \left(2 \pi \frac{x_{\max }-x}{x_{\max }-
        x_{\text {cut }}}\right) & x>x_{\text {cut }} \\
        1 & \text{else}
        \end{cases}

    Parameters
    ----------
    n: numpy.ndarray
        Array assumed to be given by np.arange(0, N // 2 + 1),
        where N is the number of elements in the array to be transformed

    n_cut: int
        Specifies where the smoothing of c_m must start

    Returns
    -------
    W: numpy.ndarray
        Array containing the smoothing coefficients
    """

    n_right = n[-1] - n_cut
    n_r = n[n[:] > n_right]
    theta_right = (n[-1] - n_r) / float(n[-1] - n_right - 1)
    W = np.ones(n.size)
    W[n[:] > n_right] = theta_right - 1 / (2 * np.pi) \
        * np.sin(2 * np.pi * theta_right)
    return W


def _g_m_vals(mu, q):
    r"""Stable _g_m_vals

    Method adapted from FAST-PT, which computes

    .. math::
        \frac{\Gamma((\mu+1+q)/2)}{\Gamma((\mu+1-q)/2)}

    Switching to asymptotic form when |Im(q)| + |mu| > cut = 200, as done
    in FAST-PT.

    Parameters
    ----------
    mu: float
        The first coefficient is related to the value of the multipole
        involved in the FFTLog integral/Hankel transform

    q: numpy.ndarray
        This array specifies the grid over which the _g_m_vals function is
        evaluated

    Returns
    -------
    g_m: numpy.ndarray
        Array containing the evaluated function
    """
    if (mu + 1 + q.real[0] == 0):
        print("gamma(0) encountered. Please change another nu value!")
        exit()
    imag_q = np.imag(q)
    g_m = np.zeros(q.size, dtype=complex)
    cut = 200
    cut_criterion_array = np.absolute(np.imag(q)) - np.absolute(mu)
    asym_q = q[cut_criterion_array > cut]
    asym_plus = (mu + 1 + asym_q) / 2.
    asym_minus = (mu + 1 - asym_q) / 2.

    q_good_bool_array = q[(cut_criterion_array <= cut) &
                          (q != mu + 1 + 0.0j)]
    q_good = q_good_bool_array

    alpha_plus = (mu + 1 + q_good) / 2.
    alpha_minus = (mu + 1 - q_good) / 2.

    g_m[(cut_criterion_array <= cut) &
        (q != mu + 1 + 0.0j)] = gamma(alpha_plus) / gamma(alpha_minus)

    # asymptotic form, taken from
    # https://github.com/JoeMcEwen/FAST-PT/blob/master/fastpt/gamma_funcs.py
    abs_im_q = np.absolute(imag_q)
    abs_mu = np.absolute(mu)

    # to improve readibility, the argument of the exponential has been divided
    # in more terms and then summed
    term1 = (asym_plus - 0.5) * np.log(asym_plus)
    term2 = - (asym_minus - 0.5) * np.log(asym_minus)
    term3 = - asym_q
    term4 = 1. / 12 * (1. / asym_plus - 1. / asym_minus)
    term5 = 1. / 360. * (1. / asym_minus**3 - 1. / asym_plus**3)
    term6 = 1 / 1260 * (1. / asym_plus**5 - 1. / asym_minus**5)
    exp_arg = term1 + term2 + term3 + term4 + term5 + term6

    g_m[cut_criterion_array > cut] = np.exp(exp_arg)

    g_m[np.where(q == mu + 1 + 0.0j)[0]] = 0. + 0.0j
    return g_m


def _g_l(ell, z_array):
    r"""Computes _g_l

    Computes the _g_l function, defined as in
    `Fang et al. (2020) <https://arxiv.org/abs/1911.11947>`

    .. math::
        g_\ell(z) = 2^z * \Gamma((\ell+z)/2) / \Gamma((3+\ell-z)/2)

    Parameters
    ----------
    ell: float
        Order of the Bessel function of the FFTLog transform
    z: numpy.ndarray
        Input array used to define the domain of _g_l

    Returns
    -------
    gl: numpy.ndarray
        Computed values of the _g_l function
    """
    gl = 2.**z_array * _g_m_vals(ell + 0.5, z_array - 1.5)
    return gl



class fftlog(object):
    """Class for the FFTLog algorithm.

    It computes integrals involving a Bessel function with the algorithm
    described in
    `Hamilton (2000) <https://arxiv.org/abs/astro-ph/9905191v4>`__ .
    It decomposes the integrand function using the Fast Fourier Transform;
    after this decomposition, each term can be integrated analytically.
    """

    def __init__(self, x, fx, nu=1.1, N_extrap_begin=0, N_extrap_end=0,
                 c_window_width=0.25, N_pad=0):
        """List of parameters.

        Parameters
        ----------
        x: numpy.ndarray
            Original logarithmically sampled domain of the input function. If
            not even, the algorithmic implementation modifies it accordingly
        fx: numpy.ndarray
            Original sampled input function
        nu: float
            Bias index, used to have a more stable fast Fourier transform
        N_extrap_begin: int
            Number of extrapolated points at the beginning of input arrays
        N_extrap_end: int
            Number of extrapolated points at the end of input arrays
        c_window_width: float
            Fraction of the ``c_m`` coefficients smoothed by the window
        N_pad: int
            Number of zero-padded points at the beginning and end of ``fx``
            array
        """
        self.x_original = x
        self.dlnx = np.log(x[1] / x[0])
        self.fx_original = fx
        self.nu = nu
        self.N_extrap_begin = N_extrap_begin
        self.N_extrap_end = N_extrap_end
        self.c_window_width = c_window_width

        # extrapolate x and f(x) linearly in log(x), and log(f(x))
        self.x = _log_extrap(x, N_extrap_begin, N_extrap_end)
        self.fx = _log_extrap(fx, N_extrap_begin, N_extrap_end)
        self.N = self.x.size  # length of the array after manipulations

        # zero-padding
        if (N_pad):
            pad = np.zeros(N_pad)
            self.x = _log_extrap(self.x, N_pad, N_pad)
            self.fx = np.hstack((pad, self.fx, pad))
            self.N += 2 * N_pad
            self.N_extrap_end += N_pad
            self.N_extrap_begin += N_pad  # update after padding

        if (self.N % 2 == 1):  # force they array size to be even, as
            self.x = self.x[:-1]  # required by the algorithm
            self.fx = self.fx[:-1]
            self.N -= 1
            if (N_extrap_end):
                self.N_extrap_end -= 1

        self.m, self.c_m = self._get_c_m()
        # array of eta_m, the exponent in the power law decomposition
        self.eta_m = 2 * np.pi * self.m / (float(self.N) * self.dlnx)

    def _get_c_m(self):
        r"""Gets smoothed coefficients.

        Computes the smoothed FFT coefficients of "biased" input
        function f(x): f_biased = f(x) / x^\nu
        (number of x values should be even).

        Returns
        -------
        Indexes m: array
            Indexes of the FFTW decomposition
        Coefficients c_m: array
            Smoothed coefficients on the biased input function
        """
        f_biased = self.fx * self.x ** (-self.nu)
        # biased f(x) array, to improve numerical stability
        c_m = rfft(f_biased)  # coefficients of the power-law decomposition
        m = np.arange(0, self.N // 2 + 1)
        c_m = c_m * _c_window(m, int(self.c_window_width * self.N // 2))
        return m, c_m

    def fftlog(self, ell):
        r"""Performs the fftlog.

        Calculates

        .. math::
            F(y) = \int_0^\infty \frac{dx}{x} f(x) j_{\rm \ell}(xy)

        where :math:`j_\ell` is the spherical Bessel function of order
        :math:`\ell`.

        Parameters
        ----------
        ell: int
            Order of the Bessel function present in the integral

        Returns
        -------
        y, Fy: numpy.ndarray, numpy.ndarray
         Logarithmically spaced array, set as
         ``y[:] = (ell+1)/x[::-1]``, FFTlog transform evaluated
         over the ``y`` array
        """
        z_ar = self.nu + 1j * self.eta_m
        y = (ell + 1.) / self.x[::-1]
        # TODO: possible improvement. y can be evaluated once and stored
        h_m = self.c_m * (self.x[0] * y[0])**(-1j * self.eta_m) \
            * _g_l(ell, z_ar)
        # TODO: possible improvement. _g_l can be evaluated once and stored

        Fy = irfft(np.conj(h_m)) * y**(-self.nu) * np.sqrt(np.pi) / 4.
        # here the ordering of N_extrap_begin and N_extrap_end is reversed
        # since we have moved to Fourier space
        return (
            y[self.N_extrap_end:self.N - self.N_extrap_begin],
            Fy[self.N_extrap_end:self.N - self.N_extrap_begin]
        )

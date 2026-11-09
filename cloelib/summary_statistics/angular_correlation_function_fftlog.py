"""Real-space angular correlation functions from angular power spectra by the
FFTLog Hankel transform, following the configuration-space implementation of
A. Spurio Mancini, which uses `mcfit`.

In the flat-sky approximation the correlation functions are Hankel
transforms of the spectra,

    xi(theta) = 1 / (2 pi) int dl l C_l J_nu(l theta),

with ``nu = 0`` for xi_+ and w(theta), ``nu = 2`` for gamma_t and ``nu = 4``
for xi_-. FFTLog (Hamilton 2000) evaluates them from C_l on a sparse
multipole grid: the integral is a convolution in log l, done with one FFT
and the analytic Mellin transform of the Bessel kernel. Unlike the full-sky
Wigner-d sum of `angular_correlation_function_wigner`, which needs every
integer multipole up to the top of the sum and loses accuracy above
l ~ 26000, it needs the spectra on a log grid of a few thousand multipoles.
"""

import numpy as np
from scipy.special import loggamma

from .angular_correlation_function_wigner import AngularCorrelationFunctionWigner

DEFAULT_ELL_MAX = 40000
DEFAULT_N_ELL = 2048
DEFAULT_TAPER = 5.0


def ell_grid(ell_max=DEFAULT_ELL_MAX, n_ell=DEFAULT_N_ELL):
    """Log-spaced multipoles from 2 to ``ell_max``, the default grid of the
    spectra behind `AngularCorrelationFunctionFFTLog`.

    Parameters:
        ell_max (float): Largest multipole.
        n_ell (int): Number of multipoles.

    Returns:
        (numpy.ndarray): The multipoles.
    """
    return np.geomspace(2.0, float(ell_max), int(n_ell))


def _mellin_bessel(z, nu):
    """Mellin transform of ``J_nu``, ``int_0^inf t^(z-1) J_nu(t) dt``, by
    analytic continuation."""
    return np.exp(
        np.log(2.0) * (z - 1) + loggamma(0.5 * (nu + z)) - loggamma(0.5 * (nu + 2 - z))
    )


def hankel_fftlog(x, f, nu, q=1.0, padding=2):
    """Hankel transform ``g(y) = int_0^inf f(x) J_nu(x y) x dx`` by FFTLog.

    Same algorithm and conventions as `mcfit.transforms.Hankel`: power-law
    bias ``q``, low-ringing output grid, and symmetric zero padding of the
    input to ``padding`` times its size (rounded up to a power of two).

    Parameters:
        x (numpy.ndarray): Log-spaced abscissae, ``x[i+1] / x[i]`` constant.
        f (numpy.ndarray): Function values on ``x``.
        nu (int): Order of the Bessel function.
        q (float): Power-law bias of the transform.
        padding (int): Zero-padding factor, 1 for none.

    Returns:
        (numpy.ndarray, numpy.ndarray): Log-spaced ordinates ``y``, with the
        spacing of ``x`` and as many as the abscissae, and the transform on
        ``y``.
    """
    x = np.asarray(x, float)
    f = np.asarray(f, float)
    n_in = x.size
    dlnx = np.log(x[-1] / x[0]) / (n_in - 1)
    n = int(2 ** np.ceil(np.log2(padding * n_in))) if padding > 1 else n_in
    left = (n - n_in) // 2
    x_pad = x[0] * np.exp((np.arange(n) - left) * dlnx)
    f_pad = np.zeros(n)
    f_pad[left : left + n_in] = f

    # Low-ringing condition: the kernel coefficient at the Nyquist frequency
    # is real, which fixes ln(x * y) of the paired abscissae and ordinates.
    lnxy = dlnx / np.pi * np.angle(_mellin_bessel(q + 1j * np.pi / dlnx, nu))
    y_pad = np.exp(lnxy) / x_pad[::-1]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dlnx)
    # The x of the Hankel measure goes with f; the bias shifts a power of x
    # from f to the kernel.
    phi = np.fft.fft(f_pad * x_pad ** (2.0 - q))
    coefficients = _mellin_bessel(q - 1j * k, nu) * np.exp(1j * k * lnxy)
    g_biased = np.roll(np.fft.ifft(np.conj(phi) * coefficients).real, -1)
    g_pad = g_biased * y_pad ** (-q)
    # Drop the padding of the output, whose ordinates pair with the reversed
    # abscissae.
    right = n - n_in - left
    return y_pad[right : right + n_in], g_pad[right : right + n_in]


class CorrelationTransform:
    """Angular power spectra on given multipoles to correlation functions at
    given separations.

    The spectra are extended to a wide log grid before the transform:
    interpolated in log-log inside the grid of the spectra, flat below it, and
    above it a falling power law with a Gaussian taper,
    ``exp(-((ell - ell_max) / (taper * ell_max))^2)``, so that the spectrum
    reaches zero well before the end of the grid (a step there rings through
    the whole transform).

    Parameters:
        ells (numpy.ndarray): Increasing multipoles of the spectra, e.g.
            `ell_grid`.
        theta (numpy.ndarray): Separations, in radians.
        theta_edges (numpy.ndarray|None): Edges of the angular bins, in
            radians, needed when ``n_sub > 1``.
        n_sub (int): Number of points per angular bin the correlation
            functions are averaged over, weighted by the area
            ``theta dtheta``; 1 evaluates them at ``theta``.
        log_bins (bool): Whether the sub-points of a bin are spaced in
            log(theta) (True) or in theta.
        taper (float): Scale of the Gaussian taper of the tail, in units of
            the largest multipole.
    """

    def __init__(
        self,
        ells,
        theta,
        theta_edges=None,
        n_sub=1,
        log_bins=True,
        taper=DEFAULT_TAPER,
    ):
        self.ells = np.asarray(ells, float)
        if (
            self.ells.ndim != 1
            or self.ells.size < 2
            or self.ells[0] <= 0
            or np.any(np.diff(self.ells) <= 0)
        ):
            raise ValueError("'ells' must be positive and strictly increasing")
        self.theta = np.asarray(theta, float)
        self._log_ells = np.log(self.ells)
        self._ells_wide = np.logspace(-2.0, 6.0, 4 * self.ells.size)
        self._above = self._ells_wide > self.ells[-1]
        self._below = self._ells_wide < self.ells[0]
        excess = np.clip(self._ells_wide - self.ells[-1], 0.0, None)
        self._taper = np.exp(-((excess / (float(taper) * self.ells[-1])) ** 2))
        self.n_sub = int(n_sub)
        if self.n_sub < 1:
            raise ValueError(f"'n_sub' must be a positive integer, got {n_sub}")
        if self.n_sub == 1:
            self._points = self.theta
            self._weights = None
            return
        if theta_edges is None:
            raise ValueError(
                "Averaging over the angular bins ('n_sub' > 1) needs their edges"
            )
        edges = np.asarray(theta_edges, float)
        if edges.size != self.theta.size + 1:
            raise ValueError(
                f"{edges.size} bin edges do not bound {self.theta.size} separations"
            )
        points, weights = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            if log_bins:
                sub = np.geomspace(low, high, self.n_sub + 1)
                points.append(np.sqrt(sub[:-1] * sub[1:]))
            else:
                sub = np.linspace(low, high, self.n_sub + 1)
                points.append(0.5 * (sub[:-1] + sub[1:]))
            area = 0.5 * (sub[1:] ** 2 - sub[:-1] ** 2)
            weights.append(area / area.sum())
        self._points = np.concatenate(points)
        self._weights = np.asarray(weights)

    def _widen(self, c_ell):
        """Spectrum on the wide grid: interpolated in log-log inside the grid
        of the spectra, flat below it and a falling power law above it."""
        c_ell = np.asarray(c_ell, float)
        log_ells_wide = np.log(self._ells_wide)
        magnitude = np.exp(
            np.interp(log_ells_wide, self._log_ells, np.log(np.abs(c_ell) + 1e-300))
        )
        sign = np.sign(np.interp(log_ells_wide, self._log_ells, c_ell))
        sign[sign == 0] = 1.0
        wide = magnitude * sign
        last, previous = abs(float(c_ell[-1])), abs(float(c_ell[-2]))
        if last > 1e-300 and previous > 1e-300:
            slope = np.log(last / previous) / (self._log_ells[-1] - self._log_ells[-2])
        else:
            slope = -3.0
        if not np.isfinite(slope):
            slope = -3.0
        # The spectra fall at high multipoles: a spurious rising tail (e.g. a
        # cross-spectrum crossing zero at the grid edge) would blow up.
        slope = float(np.clip(slope, -8.0, -0.5))
        wide[self._above] = (
            c_ell[-1] * (self._ells_wide[self._above] / self.ells[-1]) ** slope
        )
        wide[self._below] = c_ell[0]
        return wide * self._taper

    def __call__(self, c_ell, nu):
        """Correlation function of a spectrum at the separations.

        Parameters:
            c_ell (numpy.ndarray): Spectrum on `ells`.
            nu (int): Order of the Bessel function: 0 for xi_+ and w(theta),
                2 for gamma_t, 4 for xi_-.

        Returns:
            (numpy.ndarray): The correlation function, averaged over each
            angular bin when ``n_sub > 1``.
        """
        y, g = hankel_fftlog(self._ells_wide, self._widen(c_ell), nu)
        values = np.interp(self._points, y, g) / (2.0 * np.pi)
        if self._weights is None:
            return values
        return np.sum(
            values.reshape(self.theta.size, self.n_sub) * self._weights, axis=1
        )


class AngularCorrelationFunctionFFTLog(AngularCorrelationFunctionWigner):
    """Correlation function implementation using the flat-sky FFTLog Hankel
    transform.

    Drop-in alternative to `AngularCorrelationFunctionWigner`, whose tracer
    handling and output format it shares: `get_xi` returns the same
    dictionary of `TwoPointCorrelationFunction`, with xi_+ and xi_- in the
    ``[0, 0]`` and ``[1, 1]`` entries of a shear pair and gamma_t in the
    ``[0]`` entry of a positions-shear pair.

    Parameters:
        angular_two_point (AngularTwoPoint): Object providing Cl evaluation
            and tracers.
        ells (numpy.ndarray): Increasing multipoles at which the Cl are
            evaluated, e.g. `ell_grid`.
        ks (jax.numpy.ndarray): Wavenumber grid of the matter power spectrum.
        theta_edges (numpy.ndarray|None): Edges of the angular bins, in
            radians, needed when ``n_sub > 1``.
        n_sub (int): Number of points per angular bin the correlation
            functions are averaged over; 1 evaluates them at the separations.
        log_bins (bool): Whether the sub-points of a bin are spaced in
            log(theta) (True) or in theta.
        taper (float): Scale of the Gaussian taper of the spectra above the
            largest multipole, in units of that multipole.
    """

    def __init__(
        self,
        angular_two_point,
        ells,
        ks,
        theta_edges=None,
        n_sub=1,
        log_bins=True,
        taper=DEFAULT_TAPER,
    ):
        super().__init__(angular_two_point, ells, ks)
        self.theta_edges = theta_edges
        self.n_sub = n_sub
        self.log_bins = log_bins
        self.taper = taper

    def get_xi(self, theta):
        """
        Compute the angular correlation function xi(theta) by FFTLog.

        WARNING: Assumes B-modes are zero, as they are not passed on from AngularTwoPoint

        Args:
            theta (numpy.ndarray): Angles in radians.

        Returns:
            (dict): `TwoPointCorrelationFunction` of every pair of tomographic
            bins, as `AngularCorrelationFunctionWigner.get_xi`.
        """
        theta = np.asarray(theta, float)
        transform = CorrelationTransform(
            self.ells,
            theta,
            theta_edges=self.theta_edges,
            n_sub=self.n_sub,
            log_bins=self.log_bins,
            taper=self.taper,
        )
        c_ell = np.asarray(
            self.angular_two_point.get_Cl_tensor(self.ells, 0, self.ks), float
        )
        n_tomo1, n_tomo2 = c_ell.shape[1], c_ell.shape[2]
        n_theta = theta.size
        spins = (self.s1, self.s2)
        symmetric = spins in [(0, 0), (2, 2)]
        pairs = [
            (i, j)
            for i in range(n_tomo1)
            for j in (range(i, n_tomo2) if symmetric else range(n_tomo2))
        ]

        if spins == (2, 2):
            all_xi = np.zeros((2, 2, n_theta, n_tomo1, n_tomo2))
            for i, j in pairs:
                all_xi[0, 0, :, i, j] = transform(c_ell[:, i, j], 0)
                all_xi[1, 1, :, i, j] = transform(c_ell[:, i, j], 4)
        elif spins in [(2, 0), (0, 2)]:
            all_xi = np.zeros((2, n_theta, n_tomo1, n_tomo2))
            for i, j in pairs:
                all_xi[0, :, i, j] = transform(c_ell[:, i, j], 2)
        elif spins == (0, 0):
            all_xi = np.zeros((n_theta, n_tomo1, n_tomo2))
            for i, j in pairs:
                all_xi[:, i, j] = transform(c_ell[:, i, j], 0)
        else:
            raise ValueError("Spin values not as expected")

        return self.build_xi_dict(all_xi, theta, n_tomo1, n_tomo2)

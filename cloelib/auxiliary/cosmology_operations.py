"""Module for cosmological operations."""


import numpy as np
from scipy import integrate


def compute_sigma8(ks, linear_Pk):
    """Compute sigma8 from a cosmology object and its results.
    k in h/Mpc, Pk in (Mpc/h)^3
    """
    
    # Define the top-hat window function in Fourier space
    def window(k_mode):
        R = 8.0  # Mpc/h
        return 3 * (np.sin(k_mode * R) - k_mode * R * np.cos(k_mode * R)) / (k_mode * R) ** 3

    # Integrand for sigma^2
    integrand = ks**2 * linear_Pk * window(ks)**2 / (2 * np.pi**2)

    # Linear integration using trapezoidal rule
    sigma_sq = np.trapz(integrand, ks)
    return np.sqrt(sigma_sq)



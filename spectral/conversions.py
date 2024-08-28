#!/usr/bin/env python3

import matplotlib
import numpy as np
import os
import pyfftw
import pyfftw.builders as FFTW
import pyfftw.interfaces.numpy_fft as FFT

from .bm1 import M, α, β, κ, ρ, fbulk

# threaded FFTW shenanigans
pyfftw.config.NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])

def autocorrelation(data):
    """Compute the auto-correlation / 2-point statistics of a field variable"""
    signal = data - np.mean(data)
    fft = FFT.rfftn(signal)
    inv = FFT.fftshift(FFT.irfftn(fft * np.conjugate(fft)))
    cor = inv.real / (np.var(signal) * signal.size)
    return cor

def c2y(c):
    # Convert composition field to order parameter
    return (2 * c - α - β) / (β - α)


def y2c(y):
    # Convert order parameter field to composition
    return 0.5 * (β - α) * (1 + y) + α


def gamma():
    # Compute gradient energy coefficient
    return κ / (ρ * (β - α) ** 2)


def t2τ(t):
    # Nondimensionalize time
    return t / (ρ * M * (β - α) ** 2)


def τ2t(τ):
    # Dimensionalize time
    return (ρ * M * (β - α) ** 2) * τ


def free_energy(c, dx, K):
    fft = FFTW.rfftn(c.copy())
    ĉ = fft()

    fcx = FFTW.irfftn(ĉ * 1j * K[0])
    fcy = FFTW.irfftn(ĉ * 1j * K[1])

    cx = fcx().real
    cy = fcy().real

    return dx**2 * (0.5 * κ * (cx**2 + cy**2) + fbulk(c)).sum()


def log_hn(h, n, b=np.log(1000)):
    """
    Support function for plotting 𝒪(hⁿ) on a log-log scale:
      log(y) = n log(h) + b
             = log(hⁿ) + b
          y  = hⁿ exp(b)

    Inputs
    ------
    h: array of dx values
    n: order of accuracy
    b: intercept
    """
    return np.exp(b) * h**n


def radial_average(data, r, R):
    return data[(R > r - 0.5) & (R < r + 0.5)].mean()


def radial_profile(data, center=None):
    """Take the average in concentric rings around the center of a field"""
    if center is None:
        center = np.array(data.shape, dtype=int) // 2

    nx, ny = data.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    R = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = np.arange(1 + int(0.9 * center[0]))
    μ = radial_average(data, r, R)

    return r, μ


class MidpointNormalize(matplotlib.colors.Normalize):
    """
    Helper class to center the colormap on a specific value from Joe Kington
    <http://chris35wills.github.io/matplotlib_diverging_colorbar/>
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        self.vmin = vmin
        self.vmax = vmax
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # ignoring masked values and lotsa edge cases
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


class FourierInterpolant:
    """
    Spectrally-accurate reciprocal-space interpolation for field data
    on uniform rectangular grids with periodic boundary conditions.
    For derivation, see `fourier-interpolation.ipynb`.
    """

    def __init__(self, shape):
        """
        Set the "fine mesh" details
        """
        self.shape = np.array(shape, dtype=int)
        self.fine = None

        # FFT array & plan
        self.reverse = pyfftw.zeros_aligned(np.flip(shape), dtype=complex)
        self.ift = FFTW.ifftn(self.reverse)

    def pad(self, ŵ):
        """
        Zero-pad "before and after" coarse data to fit fine mesh size
        in 1D or 2D (3D untested), with uniform rectangular grids

        Input
        -----
        ŵ -- Fourier-transformed coarse field data to pad
        """
        M = np.flip(self.shape)  # FFT rotates the array
        N = np.array(ŵ.shape)
        P = np.subtract(M, N, dtype=int) // 2  # ≡ (M - N) // 2
        P = P.reshape((len(N), 1))
        return np.pad(ŵ, P)

    def upsample(self, w):
        """
        Interpolate the coarse field data $w$ onto the fine mesh
        """
        fft = FFTW.fftn(w.copy())
        ŵ = FFT.fftshift(fft())

        û = self.pad(ŵ)
        self.reverse[:] = FFT.ifftshift(û)
        scale = np.prod(np.array(û.shape)) / np.prod(np.array(w.shape))
        return scale * self.ift().real

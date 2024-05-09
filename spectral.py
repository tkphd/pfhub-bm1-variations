#!/usr/bin/env python3

import numpy as np
import matplotlib
# from line_profiler import profile

π = np.pi
L = 200

α = 0.3  # eqm composition of phase A
β = 0.7  # eqm composition of phase B
ρ = 5.0  # well height
κ = 2.0  # gradient energy coeff
M = 5.0  # diffusivity


class MidpointNormalize(matplotlib.colors.Normalize):
    """
    Helper class to center the colormap on a specific value from Joe Kington
    <http://chris35wills.github.io/matplotlib_diverging_colorbar/>
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # ignoring masked values and lotsa edge cases
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# @profile
def finterf(c_hat, Ksq):
    # interfacial free energy density
    return κ * np.fft.irfftn(Ksq * c_hat**2).real


# @profile
def fbulk(c):
    # bulk free energy density
    return ρ * (c - α)**2 * (β - c)**2


# @profile
def dfdc(c):
    # derivative of bulk free energy density
    return 2 * ρ * (c - α) * (β - c) * (α + β - 2 * c)


# @profile
def dfdc_linear(c):
    return 2 * ρ * ((α**2 + 2 * α * β + β**2) * c - α**2 * β - α * β**2)

# @profile
def dfdc_nonlinear(c):
    return 4.0 * ρ * c**3 - 6.0 * ρ * (α + β) * c**2


# @profile
def c_x(c_hat, K):
    return np.fft.irfftn(c_hat * 1j * K[0]).real


# @profile
def c_y(c_hat, K):
    return np.fft.irfftn(c_hat * 1j * K[1]).real


# @profile
def free_energy(c, c_hat, K, dx):
    """
    Cf. Trefethen Eqn. 12.5: typical integration is sub-spatially
    accurate, but this trapezoid rule retains accuracy.
    """
    cx = c_x(c_hat, K)
    cy = c_y(c_hat, K)
    return dx**2 * (κ/2 * (cx**2 + cy**2) + fbulk(c)).sum()


# @profile
def autocorrelation(data):
    """Compute the auto-correlation / 2-point statistics of a field variable"""
    signal = data - np.mean(data)
    fft = np.fft.rfftn(signal)
    inv = np.fft.fftshift(np.fft.irfftn(fft * np.conjugate(fft)))
    # cor = np.fft.ifftshift(inv).real / (np.var(signal) * signal.size)
    cor = inv.real / (np.var(signal) * signal.size)
    return cor


# @profile
def radial_average(data, r, R):
    return data[(R > r - 0.5) & (R < r + 0.5)].mean()


# @profile
def radial_profile(data, center=None):
    """Take the average in concentric rings around the center of a field"""
    if center is None:
        center = np.array(data.shape, dtype=int) // 2

    nx, ny = data.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    R = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = np.arange(center[0]+1)
    ravg = lambda r: data[(R > r - 0.5) & (R < r + 0.5)].mean()
    μ = np.vectorize(ravg)(r)

    return r, μ


class Evolver:
    # @profile
    def __init__(self, c, c_old, dx):
        self.dx = dx

        self.c = c.copy()
        self.c_old = c_old.copy()
        self.c_sweep = np.ones_like(self.c)

        self.c_hat = np.fft.rfftn(self.c)
        self.c_hat_prev = np.ones_like(self.c_hat)
        self.c_hat_old = self.c_hat.copy()

        self.dfdc_hat = np.ones_like(self.c_hat)

        # prepare auxiliary arrays
        kx = 2.0 * π * np.fft.fftfreq(self.c.shape[0], d=self.dx)
        ky = 2.0 * π * np.fft.rfftfreq(self.c.shape[1], d=self.dx)
        self.K = np.array(np.meshgrid(kx, ky, indexing="ij"), dtype=np.float64)
        self.Ksq = np.sum(self.K * self.K, axis=0, dtype=np.float64)

        # coefficient of terms linear in c_hat
        self.linear_coefficient = \
            2.0 * ρ * (α**2 + 4.0 * α * β + β**2) - α**2 * β - α * β**2 + κ * self.Ksq

        # # dealias the flux capacitor
        # self.nyquist_mode = kx.max() / 2
        # self.alias_mask = np.array( (np.abs(self.K[0]) < self.nyquist_mode) \
        #                           * (np.abs(self.K[1]) < self.nyquist_mode),
        #                             dtype=bool)

    # @profile
    def free_energy(self):
        return free_energy(self.c, self.c_hat, self.K, self.dx)

    # @profile
    def residual(self, numer_coeff, denom_coeff):
        # r = F(xⁿ)
        return np.abs(np.linalg.norm((1.0 - denom_coeff) * self.c_hat - numer_coeff * self.dfdc_hat)).real

    # @profile
    def sweep(self, numer_coeff, denom_coeff):
        # Always sweep the non-linear terms at least twice
        self.c_hat_prev[:] = self.c_hat

        # self.dfdc_hat[:] = self.alias_mask * np.fft.rfftn(dfdc_nonlinear(self.c_sweep))
        self.dfdc_hat[:] = np.fft.rfftn(dfdc_nonlinear(self.c_sweep))

        self.c_hat[:] = \
            (self.c_hat_old - numer_coeff * self.dfdc_hat) / denom_coeff

        self.c[:] = np.fft.irfftn(self.c_hat).real

        self.c_sweep[:] = self.c

    # @profile
    def solve(self, dt, maxsweeps=20, rtol=1e-6):
        # semi-implicit discretization of the PFHub equation of motion
        residual = 1.0
        sweep = 0

        # take a stab at the "right" solution
        # Thanks to @reid-a for contributing this idea!
        self.c_sweep[:] = 2.0 * self.c - self.c_old  # reasonable guess via Taylor expansion

        self.c_hat_old[:] = self.c_hat  # required (first term on r.h.s.)
        self.c_old[:] = self.c

        numer_coeff = dt * M * self.Ksq  # used in the numerator
        denom_coeff = 1.0 + dt * M * self.Ksq * self.linear_coefficient

        # iteratively update c in place
        for _ in range(3):
            self.sweep(numer_coeff, denom_coeff)
            sweep += 1

        residual = self.residual(numer_coeff, denom_coeff)

        while residual > rtol and sweep < maxsweeps:
            self.sweep(numer_coeff, denom_coeff)
            residual = self.residual(numer_coeff, denom_coeff)
            sweep += 1

        if sweep >= maxsweeps:
            raise ValueError(f"Exceeded {maxsweeps} sweeps with res = {residual}")

        return residual, sweep


class FourierInterpolant:
    """
    Spectrally-accurate reciprocal-space interpolation for field data
    on uniform rectangular grids with periodic boundary conditions.
    For derivation, see `fourier-interpolation.ipynb`.
    """
    # @profile
    def __init__(self, shape):
        """
        Set the "fine mesh" details
        """
        self.shape = shape
        self.fine = None

    # @profile
    def pad(self, v_hat):
        """
        Zero-pad "before and after" coarse data to fit fine mesh size
        in 1D or 2D (3D untested), with uniform rectangular grids

        Input
        -----
        v_hat -- Fourier-transformed coarse field data to pad
        """
        M = np.flip(self.shape)  # transformation rotates the mesh
        N = np.array(v_hat.shape)
        z = np.subtract(M, N, dtype=int) // 2  # ≡ (M - N) // 2
        z = z.reshape((len(N), 1))
        return np.pad(v_hat, z)

    # @profile
    def upsample(self, v):
        """
        Interpolate the coarse field data $v$ onto the fine mesh
        """
        v_hat = np.fft.fftshift(np.fft.fftn(v))
        u_hat = self.pad(v_hat)
        scale = np.prod(np.array(u_hat.shape)) / np.prod(np.array(v.shape))
        return scale * np.fft.ifftn(np.fft.ifftshift(u_hat)).real


# @profile
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


# @profile
def progression(start=0):
    """
    Generate a sequence of numbers that progress in logarithmic space:
    1, 2,.. 10, 20,.. 100, 200,.. 1000, 2000, etc.
    but *don't* store them all in memory!

    Thanks to @reid-a for contributing this generator.
    """
    if start == 0:
        value = 0
        delta = 1
    else:
        """
        When progression() is called, it will increment value,
        so we have to under-shoot
        """
        value = 10**np.ceil(np.log10(start)).astype(int)
        delta = 10**np.floor(np.log10(start)).astype(int) if value < 10_000 else 1000

        while value > start:
            value -= delta
        print(f"Δ = {delta}, t = {value}")

    while True:
        value += delta
        yield value
        if (value < 10_000) and (value == 10 * delta):
            delta = value
        elif (value == 100_000):
            delta *= 10

#!/usr/bin/env python3

import matplotlib
import numpy as np
# import numpy.fft as FFT
import os
import pyfftw
import pyfftw.builders as FFTW
import pyfftw.interfaces.numpy_fft as FFT

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


def finterf(c_hat, Ksq):
    # interfacial free energy density
    return κ * FFT.irfftn(Ksq * c_hat**2).real


def fbulk(c):
    # bulk free energy density
    return ρ * (c - α)**2 * (β - c)**2


def dfdc(c):
    # derivative of bulk free energy density
    # cf. TK_R6_p551
    return 2 * ρ * (c - α) * (β - c) * (α + β - 2 * c)


def dfdc_lin(c):
    # derivative of bulk free energy density (contractive/convex part)
    return 2 * ρ * (α**2 + 4 * α * β + β**2) * c


def dfdc_exc(c):
    # derivative of bulk free energy density (expansive/concave part)
    return 2 * ρ * (2 * c**3 - 3 * (α + β) * c**2 - (α**2 * β + α * β**2))


def autocorrelation(data):
    """Compute the auto-correlation / 2-point statistics of a field variable"""
    signal = data - np.mean(data)
    fft = FFT.rfftn(signal)
    inv = FFT.fftshift(FFT.irfftn(fft * np.conjugate(fft)))
    cor = inv.real / (np.var(signal) * signal.size)
    return cor


def radial_average(data, r, R):
    return data[(R > r - 0.5) & (R < r + 0.5)].mean()


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
    def __init__(self, c, c_old, dx):
        sc = list(c.shape)
        sk = list(sc)
        sk[-1] = 1 + sk[-1] // 2

        self.dx = dx
        pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
        if "OMP_NUM_THREADS" in os.environ.keys():
            pyfftw.config.NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
        else:
            pyfftw.config.NUM_THREADS = 1

        self.a2 = 0.2  # degree of explicitness

        # allocate spatial arrays
        self.c     = pyfftw.zeros_aligned(sc)
        self.c_old = pyfftw.zeros_aligned(sc)
        self.c_tmp = pyfftw.zeros_aligned(sc)

        self.u     = pyfftw.zeros_aligned(sc)
        self.u_lin = pyfftw.zeros_aligned(sc)
        self.u_exc = pyfftw.zeros_aligned(sc)

        # assign field values
        self.c[:]  = c
        self.u[:]  = dfdc(self.c)
        self.c_old[:] = c_old

        # reciprocal arrays
        fft_c      = FFTW.rfft2(self.c.copy())
        self.ĉ     = pyfftw.zeros_aligned(sk, dtype=complex)
        self.ĉ[:]  = fft_c()

        fft_o         = FFTW.rfft2(self.c_old.copy())
        self.ĉ_old    = pyfftw.zeros_aligned(sk, dtype=complex)
        self.ĉ_old[:] = fft_o()

        fft_u      = FFTW.rfft2(self.u.copy())
        self.û     = pyfftw.zeros_aligned(sk, dtype=complex)
        self.û[:]  = fft_u()

        self.û_lin = pyfftw.zeros_aligned(sk, dtype=complex)
        self.û_exc = pyfftw.zeros_aligned(sk, dtype=complex)

        # crunch auxiliary variables
        kx = 2.0 * π * FFT.fftfreq(sc[0], d=self.dx)
        ky = 2.0 * π * FFT.rfftfreq(sc[1], d=self.dx)
        self.K = np.array(np.meshgrid(kx, ky, indexing="ij"))
        self.Ksq = np.sum(self.K * self.K, axis=0)
        self.K4 = self.Ksq**2

    def free_energy(self):
        fcx = FFTW.irfftn(self.ĉ * 1j * self.K[0])
        fcy = FFTW.irfftn(self.ĉ * 1j * self.K[1])

        cx = fcx().real
        cy = fcy().real

        return self.dx**2 * (κ/2 * (cx**2 + cy**2) + fbulk(self.c)).sum()

    def residual(self, numer_coeff, denom_coeff):
        self.u[:] = dfdc(self.c)
        fft_u = FFTW.rfft2(self.u.copy())
        self.û[:] = fft_u()

        return np.linalg.norm(
            self.ĉ_old - numer_coeff * self.û - denom_coeff * self.ĉ
        ).real

    def sweep(self, numer_coeff, denom_coeff):
        self.u_exc[:] = dfdc_exc(self.c)
        fft_exc = FFTW.rfft2(self.u_exc.copy())
        self.û_exc[:] = fft_exc()

        self.u_lin[:] = dfdc_lin(self.c)
        fft_lin = FFTW.rfft2(self.u_lin.copy())
        self.û_lin[:] = fft_lin()

        self.ĉ[:] = (self.ĉ_old - numer_coeff * (self.û_exc + self.a2 * self.û_lin)) / denom_coeff

        ift_c = FFTW.irfft2(self.ĉ.copy())
        self.c[:] = ift_c()


    def evolve(self, dt, maxsweeps=10, rtol=1e-6):
        # semi-implicit discretization of the PFHub equation of motion
        swe = 0
        res = 1.0

        A_lin = dfdc_lin(1.0)
        numer_coeff = M * dt * self.Ksq  # used in the numerator
        denom_coeff = 1 + numer_coeff * ((1 - self.a2) * A_lin +  κ * self.Ksq)

        # Make a reasonable guess at the "right" solution via Taylor expansion
        # N.B.: Compute initial guess before updating c_old!
        # Thanks to @reid-a for contributing this idea!
        self.c_tmp[:] = 2 * self.c - self.c_old

        self.c_old[:] = self.c
        self.ĉ_old[:] = self.ĉ

        self.c[:] = self.c_tmp

        # iteratively update c in place, updating non-linear coefficients
        while res > rtol and swe < maxsweeps:
            self.sweep(numer_coeff, denom_coeff)
            res = self.residual(numer_coeff, denom_coeff)
            swe += 1
            if swe == 1 or (swe % 5) == 0 or swe == maxsweeps:
                print(f"    sweep {swe:2d}: η={res:,}")

        if swe >= maxsweeps:
            raise ValueError(f"Exceeded {maxsweeps:,} sweeps with res = {res:,}")

        return res, swe


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
        self.shape = shape
        self.fine = None

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

    def upsample(self, v):
        """
        Interpolate the coarse field data $v$ onto the fine mesh
        """
        v_hat = FFT.fftshift(FFT.fftn(v))
        u_hat = self.pad(v_hat)
        scale = np.prod(np.array(u_hat.shape)) / np.prod(np.array(v.shape))
        return scale * FFT.ifftn(FFT.ifftshift(u_hat)).real


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

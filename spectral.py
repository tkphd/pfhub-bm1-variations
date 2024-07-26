#!/usr/bin/env python3

import matplotlib
import numpy as np
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
    ĉ = Ksq * c_hat**2
    ift_c = FFTW.irfft2(ĉ)
    return κ * ift_c().real


def fbulk(c):
    # bulk free energy density
    return ρ * (c - α)**2 * (β - c)**2


def dfdc(c):
    # derivative of bulk free energy density
    # cf. TK_R6_p551
    return 2 * ρ * (c - α) * (β - c) * (α + β - 2 * c)


def dfdc_lin(c):
    # derivative of bulk free energy density (linnear term)
    return 2 * ρ * (α**2 + 4 * α * β + β**2) * c


def dfdc_nln(c0, c):
    # derivative of bulk free energy density (non-linear terms)
    #      (4 * ρ * c  - 6 * ρ * (α + β)) * c  * c - 2 * ρ * (α**2 * β + α * β**2)
    return (4 * ρ * c0 - 6 * ρ * (α + β)) * c0 * c - 2 * ρ * (α**2 * β + α * β**2)


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
        self.dx = dx

        sc = list(c.shape)
        sk = list(sc)
        sk[-1] = 1 + sk[-1] // 2

        pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
        if "OMP_NUM_THREADS" in os.environ.keys():
            pyfftw.config.NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
        else:
            pyfftw.config.NUM_THREADS = 1

        # auxiliary variables
        kx = 2.0 * π * FFT.fftfreq(sc[0], d=self.dx)
        ky = 2.0 * π * FFT.rfftfreq(sc[1], d=self.dx)

        self.K = np.array(np.meshgrid(kx, ky, indexing="ij"))
        self.Ksq = np.sum(self.K * self.K, axis=0)

        self.nyquist = np.sqrt(np.amax(self.Ksq)) # Nyquist mode
        # self.dealias = pyfftw.byte_align(
        #     np.array((1.5 * self.K[0] < self.nyquist) * (1.5 * self.K[1] < self.nyquist)
        #              , dtype=bool))

        # spatial arrays
        self.c     = pyfftw.zeros_aligned(sc)
        self.c_old = pyfftw.zeros_aligned(sc)
        self.c_sweep = pyfftw.zeros_aligned(sc)

        # FFT arrays & Plans
        self.forward = pyfftw.zeros_aligned(sc)
        self.reverse = pyfftw.zeros_aligned(sk, dtype=complex)

        self.fft = FFTW.rfft2(self.forward)
        self.ift = FFTW.irfft2(self.reverse)

        # reciprocal arrays
        self.ĉ     = pyfftw.zeros_aligned(sk, dtype=complex)
        self.ĉ_old = pyfftw.zeros_aligned(sk, dtype=complex)
        self.ĉ_num = pyfftw.zeros_aligned(sk, dtype=complex)  # quicker residuals
        self.û     = pyfftw.zeros_aligned(sk, dtype=complex)

        # assign field values
        self.c[:]     = c
        self.c_old[:] = c_old

        self.forward[:] = self.c
        self.ĉ[:]       = self.fft()

        self.forward[:] = self.c_old
        self.ĉ_old[:]   = self.fft()

        self.forward[:] = dfdc(self.c)
        self.û[:]       = self.fft()


    def free_energy(self):
        fcx = FFTW.irfft2(self.ĉ * 1j * self.K[0])
        fcy = FFTW.irfft2(self.ĉ * 1j * self.K[1])

        cx = fcx().real
        cy = fcy().real

        return self.dx**2 * (κ/2 * (cx**2 + cy**2) + fbulk(self.c)).sum()


    def mass(self):
        dV = self.dx**2
        return dV * np.sum(self.c)

    def residual(self, new_co):
        return np.linalg.norm(
            self.ĉ_num - new_co * self.ĉ
        ).real


    def sweep(self, old_co, new_co):
        self.forward[:] = dfdc_nln(self.c_sweep, self.c)
        # self.û[:] = self.dealias * self.fft()
        self.û[:] = self.fft()

        self.ĉ_num[:] = self.ĉ_old - old_co * self.û
        self.ĉ[:] = self.ĉ_num / new_co

        self.c_sweep[:] = self.c

        self.reverse[:] = self.ĉ
        self.c[:] = self.ift()

        return self.residual(new_co)


    def evolve(self, dt, maxsweeps=20, residual_tolerance=1e-12, convergence_tolerance=1e-3):
        # semi-implicit discretization of the PFHub equation of motion
        l2c = 1.0
        res = 1.0
        swe = 0

        old_co = M * dt * self.Ksq  # used in the numerator
        new_co = 1 + old_co * (dfdc_lin(1) + κ * self.Ksq)

        # Make a reasonable guess at the "right" solution via Taylor expansion
        # N.B.: Compute initial guess before updating c_old!
        # Thanks to @reid-a for contributing this idea!
        self.c_sweep[:] = self.c  # previous "guess" for the swept solution
        self.c[:] = 2 * self.c - self.c_old  # latest approx of the solution

        # c holds a better guess of the solution at the new time,
        # c_sweep holds the previous guess. The two should converge with sweeping.

        # iteratively update c in place, updating non-linear coefficients
        while (res > residual_tolerance or l2c > convergence_tolerance) and swe < maxsweeps:
            res = self.sweep(old_co, new_co)
            l2c = np.linalg.norm(self.c - self.c_sweep)
            swe += 1

        self.c_old[:] = self.c
        self.ĉ_old[:] = self.ĉ

        if swe >= maxsweeps and res > residual_tolerance:
            raise ValueError(f"Exceeded {maxsweeps:,} sweeps with res = {res:,}")

        return swe, res, l2c


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
        delta = 10**np.floor(np.log10(start)).astype(int)

        while value > start:
            value -= delta
        print(f"Δ = {delta}, t = {value}")

    while True:
        value += delta
        yield value
        if value == 10 * delta:
            delta = value

#!/usr/bin/env python3

import numpy as np

π = np.pi
L = 200

α = 0.3  # eqm composition of phase A
β = 0.7  # eqm composition of phase B
ρ = 5.0  # well height
κ = 2.0  # gradient energy coeff
M = 5.0  # diffusivity

def finterf(c_hat, Ksq):
    # interfacial free energy density
    return κ * np.fft.ifft2(Ksq * c_hat**2).real


def fbulk(c):
    # bulk free energy density
    return ρ * (c - α)**2 * (β - c)**2


def dfdc(c):
    # derivative of bulk free energy density
    return 2 * ρ * (c - α) * (β - c) * (α + β - 2 * c)


def dfdc_nonlinear(c):
    return 2 * ρ * (2 * c**3 - 3 * (α + β) * c**2 - α**2 * β - α * β**2)


def c_x(c_hat, K):
    return np.fft.ifft2(c_hat * 1j * K[0]).real


def c_y(c_hat, K):
    return np.fft.ifft2(c_hat * 1j * K[1]).real


def free_energy(c, c_hat, K, dx):
    """
    Cf. Trefethen Eqn. 12.5: typical integration is sub-spatially
    accurate, but this trapezoid rule retains accuracy.
    """
    c_x_hat = c_x(c_hat, K)
    c_y_hat = c_y(c_hat, K)
    return dx**2 * (κ/2 * (c_x_hat**2 + c_y_hat**2) + fbulk(c)).sum()


class Evolver:
    def __init__(self, c, c_old, dx):
        self.dx = dx

        self.c = c.copy()
        self.c_old = c_old.copy()
        self.c_sweep = np.ones_like(self.c)

        self.c_hat = np.fft.fft2(self.c)
        self.c_hat_prev = np.ones_like(self.c_hat)
        self.c_hat_old = self.c_hat.copy()

        self.dfdc_hat = np.ones_like(self.c_hat)

        # prepare auxiliary arrays
        k = 2 * π * np.fft.fftfreq(self.c.shape[0], d=self.dx)
        self.K = np.array(np.meshgrid(k, k, indexing="ij"), dtype=np.float64)
        self.Ksq = np.sum(self.K * self.K, axis=0, dtype=np.float64)

        # coefficient of terms linear in c_hat
        self.linear_coefficient = 2 * ρ * (α**2 + 4 * α * β + β**2) \
                                + κ * self.Ksq

        # dealias the flux capacitor
        self.nyquist_mode = 2.0 * k.max() / 3
        self.alias_mask = np.array( (np.abs(self.K[0]) < self.nyquist_mode) \
                                  * (np.abs(self.K[1]) < self.nyquist_mode),
                                    dtype=bool)


    def free_energy(self):
        return free_energy(self.c, self.c_hat, self.K, self.dx)


    def residual(self, numer_coeff, denom_coeff):
        return np.linalg.norm(np.abs(self.c_hat_old - numer_coeff * self.dfdc_hat
                              - denom_coeff * self.c_hat_prev).real)


    def sweep(self, numer_coeff, denom_coeff):
        self.c_hat_prev[:] = self.c_hat

        self.dfdc_hat[:] = self.alias_mask * np.fft.fft2(dfdc_nonlinear(self.c_sweep))

        self.c_hat[:] = (self.c_hat_old - numer_coeff * self.dfdc_hat) / denom_coeff

        self.c[:] = np.fft.ifft2(self.c_hat).real

        return self.residual(numer_coeff, denom_coeff)


    def solve(self, dt):
        # semi-implicit discretization of the PFHub equation of motion
        residual = 1.0
        sweep = 0

        # take a stab at the "right" solution
        # Thanks to @reid-a for contributing this idea!
        self.c_sweep[:] = 2 * self.c - self.c_old  # reasonable guess

        self.c_hat_old[:] = self.c_hat  # required (first term on r.h.s.)
        self.c_old[:] = self.c

        numer_coeff = dt * M * self.Ksq  # used in the numerator
        denom_coeff = 1 + dt * M * self.Ksq * self.linear_coefficient # denominator

        # iteratively update c in place
        while sweep < 1000 and residual > 1e-3:
            residual = self.sweep(numer_coeff, denom_coeff)

            if not np.isfinite(residual):
                raise ValueError("Residual is NAN!")

            self.c_sweep[:] = self.c

            sweep += 1

        return residual, sweep


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
        N = v_hat.shape
        z = np.subtract(M, N, dtype=int) // 2  # ≡ (M - N) // 2
        z = z.reshape((len(N), 1))
        return np.pad(v_hat, z)


    def upsample(self, v):
        """
        Interpolate the coarse field data $v$ onto the fine mesh
        """
        v_hat = np.fft.fftshift(np.fft.fftn(v))
        u_hat = self.pad(v_hat)
        scale = np.prod(np.array(u_hat.shape)) / np.prod(np.array(v.shape))
        return scale * np.fft.ifftn(np.fft.ifftshift(u_hat)).real

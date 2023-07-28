#!/usr/bin/env python3

import numpy as np
import numpy.linalg as LA

α = 0.3  # eqm composition of phase A
β = 0.7  # eqm composition of phase B
ρ = 5.0  # well height
κ = 2.0  # gradient energy coeff
M = 5.0  # diffusivity

def finterf(c_hat, Ksq):
    # interfacial free energy density
    return κ * np.fft.ifft2(Ksq * c_hat**2)


def fbulk(c):
    # bulk free energy density
    return ρ * (c - α)**2 * (c - β)**2


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
    return dx**2 * (κ/2 * (c_x(c_hat, K)**2 + c_y(c_hat, K)**2) + fbulk(c)).sum()


class Evolver:
    def __init__(self, c, dx):
        self.dx = dx

        self.c = c.copy()
        self.c_hat = np.fft.fft2(self.c)
        self.c_hat_prev = self.c_hat.copy()

        self.c_old = self.c.copy()
        self.c_hat_old = np.empty_like(self.c_hat)

        self.c_sweep = np.empty_like(self.c)

        self.dfdc_hat = np.empty_like(self.c_hat)

        # prepare auxiliary arrays
        k = 2 * np.pi * np.fft.fftfreq(self.c.shape[0], d=self.dx)
        self.K = np.array(np.meshgrid(k, k, indexing="ij"), dtype=float)
        self.Ksq = np.sum(self.K * self.K, axis=0, dtype=float)

        # coefficient of terms linear in c_hat
        self.linear_coefficient = 2 * ρ * (α**2 + 4 * α * β + β**2) \
                                + κ * self.Ksq

        # dealias the flux capacitor
        self.nyquist_mode = 2.0 * k.max() / 3
        self.alias_mask = np.where(
            np.sqrt(self.K[0]**2 + self.K[1]**2) < self.nyquist_mode,
            True, False).astype(bool)


    def free_energy(self):
        return free_energy(self.c, self.c_hat, self.K, self.dx)


    def residual(self, numer_coeff, denom_coeff):
        return LA.norm(np.abs(self.c_hat_old - numer_coeff * self.dfdc_hat
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
        while sweep < 200 and residual > 1e-3:
            residual = self.sweep(numer_coeff, denom_coeff)

            if not np.isfinite(residual):
                raise ValueError("Residual is NAN!")

            self.c_sweep[:] = self.c

            sweep += 1

        return self.free_energy(), residual, sweep

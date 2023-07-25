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
    return 2 * ρ * (c - α) * (c - β) * (2 * c - (α + β))

def dfdc_split(c_new, c_old):
    # derivative of bulk free energy density
    return 2 * ρ * (c_old - α) * (c_old - β) * (2 * c_new - (α + β))


def c_x(c_hat, K):
    return np.fft.ifft2(c_hat * 1j * K[0])


def c_y(c_hat, K):
    return np.fft.ifft2(c_hat * 1j * K[1])


def free_energy(c, c_hat, K, dx):
    return (0.5 * κ * (c_x(c_hat, K)**2 + c_y(c_hat, K)**2).real + fbulk(c)).sum() * dx**2


class Evolver:
    def __init__(self, c, dx, sweeps):
        self.dx = dx
        self.sweeps = sweeps

        self.c = c.copy()
        self.c_old = np.empty_like(self.c)

        self.c_hat = np.fft.fft2(self.c)
        self.c_hat_old = np.empty_like(self.c_hat)

        self.dfdc_hat = np.empty_like(self.c_hat)

        # prepare auxiliary arrays
        k = 2 * np.pi * np.fft.fftfreq(self.c.shape[0], d=self.dx)
        self.K = np.array(np.meshgrid(k, k, indexing="ij"))
        self.Ksq = np.sum(self.K * self.K, axis=0)

        # dealias the flux capacitor
        self.nyquist_mode = 2 * k.max() / 3
        self.alias_mask = np.array(np.sqrt(self.K[0]**2 + self.K[1]**2) < self.nyquist_mode,
                                  dtype=bool)

    # def solve(self, dt):
    #     # iterate with c_hat pinned to old value
    #     self.c_old[:] = self.c
    #     self.c_hat_old[:] = self.c_hat
    #     res = np.zeros(self.sweeps)

    #     for sweep in range(self.sweeps):
    #         self.dfdc_hat[:] = np.where(self.alias_mask,
    #                                     np.fft.fft2(dfdc(self.c)),
    #                                     0.)
    #         self.c_hat[:] = (self.c_hat_old - dt * self.Ksq * M * self.dfdc_hat) \
    #                       / (1 + dt * M * κ * self.Ksq**2)
    #         self.c[:] = np.abs(np.fft.ifft2(self.c_hat)).astype(float)

    #         res[sweep] = LA.norm(self.c - self.c_old)

    #     return self.free_energy(), res

    def solve(self, dt):
        # iterate with c_hat pinned to old value, and
        # naïvely splitting dfdc_hat between old and new values
        self.c_old[:] = self.c
        self.c_hat_old[:] = self.c_hat
        res = np.empty(self.sweeps, dtype=float)

        delta = 1.0
        sweep = 0

        while sweep < self.sweeps and delta > 1e-5:
            self.dfdc_hat[:] = np.where(self.alias_mask,
                                        np.fft.fft2(dfdc_split(self.c, self.c_old)),
                                        0.)
            self.c_hat[:] = (self.c_hat_old - dt * self.Ksq * M * self.dfdc_hat) \
                          / (1 + dt * M * κ * self.Ksq**2)
            self.c[:] = np.abs(np.fft.ifft2(self.c_hat)).astype(float)

            res[sweep] = LA.norm(self.c - self.c_old)

            if sweep > 1:
                delta = np.abs(res[sweep] - res[sweep - 1]) / res[sweep - 1]

            sweep += 1

        return self.free_energy(), res


    def free_energy(self):
        return free_energy(self.c, self.c_hat, self.K, self.dx)

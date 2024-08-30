#!/usr/bin/env python3

import numpy as np
import os
import pyfftw
import pyfftw.builders as FFTW
import pyfftw.interfaces.numpy_fft as FFT

from .bm1 import M, α, β, κ, ρ

# threaded FFTW shenanigans
pyfftw.config.NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])

def c2y(c):
    # Convert composition field to order parameter
    return (2 * c - α - β) / (β - α)

def y2c(y):
    # Convert order parameter field to composition
    return 0.5 * (β - α) * (1 + y) + α

def gamma():
    # Compute gradient energy coefficient
    return κ / (ρ * (β - α) ** 2)

def τ2t(τ):
    # Nondimensionalize time
    return τ / (ρ * M * (β - α) ** 2)

def t2τ(t):
    # Dimensionalize time
    return (ρ * M * (β - α) ** 2) * t

class CahnHilliardEvolver:
    def __init__(self, y, y_old, dx, γ, a1=3, a2=0):
        sy = list(y.shape)
        sk = list(sy)
        sk[-1] = 1 + sk[-1] // 2

        self.dx = dx
        self.dV = dx ** len(sy)
        self.V = float(self.dV * np.prod(sy))

        self.γ = γ
        self.a1 = a1
        self.a2 = a2
        self.a1c = 1 - self.a1
        self.a2c = 1 - self.a2

        # auxiliary variables
        kx = 2 * np.pi * FFT.fftfreq(sy[0], d=self.dx)
        ky = 2 * np.pi * FFT.rfftfreq(sy[1], d=self.dx)
        k = np.array(np.meshgrid(kx, ky, indexing="ij"))

        # improved scaling in k-space, thanks to Nana Ofori-Opoku
        self.K = np.array([0.5 * (1 - np.cos(k[0])) * (3 + np.cos(k[1])),
                           0.5 * (1 - np.cos(k[1])) * (3 + np.cos(k[0]))])
        self.Ksq = np.sum(self.K * self.K, axis=0)

        self.nyquist = np.sqrt(self.Ksq.max())  # Nyquist mode
        self.dealias = pyfftw.byte_align(
            np.array(
                (1.5 * self.K[0] < self.nyquist) * (1.5 * self.K[1] < self.nyquist),
                dtype=bool,
            )
        )

        # spatial arrays
        self.y = pyfftw.zeros_aligned(sy)
        self.y_old = pyfftw.zeros_aligned(sy)

        # FFT arrays & Plans
        self.forward = pyfftw.zeros_aligned(sy)
        self.reverse = pyfftw.zeros_aligned(sk, dtype=complex)

        self.fft = FFTW.rfftn(self.forward)
        self.ift = FFTW.irfftn(self.reverse)

        # reciprocal arrays
        self.ŷ = pyfftw.zeros_aligned(sk, dtype=complex)
        self.ŷ_old = pyfftw.zeros_aligned(sk, dtype=complex)
        self.û = pyfftw.zeros_aligned(sk, dtype=complex)

        # assign field values
        self.y[:] = y
        self.y_old[:] = y_old

        self.forward[:] = self.y
        self.ŷ[:] = self.fft()


    def fbulk(self):
        # bulk free energy density
        return 0.25 * (self.y**2 - 1)**2

    def dfdc(self):
        # derivative of bulk free energy density
        # cf. TK_R6_p551
        return self.y * (self.y**2 - 1)

    def dfdc_nln(self):
        # nonlinear terms
        return self.y**3

    def _free_energy(self):
        # free energy of φ
        fcx = FFTW.irfftn(self.ŷ * 1j * self.K[0])
        fcy = FFTW.irfftn(self.ŷ * 1j * self.K[1])

        fgrad = np.float_power(fcx().real, 2) + np.float_power(fcy().real, 2)

        return self.dx**2 * (0.5 * self.γ * fgrad + self.fbulk()).sum()

    def energy_density(self):
        return self._free_energy() / self.V

    def _mass(self):
        # "mass" of φ
        return self.dV * np.sum(self.y)

    def evolve(self, dτ):
        self.y_old[:] = self.y
        self.ŷ_old[:] = self.ŷ

        self.forward[:] = self.dfdc_nln()
        self.û[:] = self.dealias * self.fft()

        dk = dτ * self.Ksq
        lin_coef = 1 + dk * (self.a1 - self.γ * self.Ksq * self.a2)
        lhs_coef = 1 - dk * (self.a1c - self.γ * self.Ksq * self.a2c)

        self.ŷ[:] = (lin_coef * self.ŷ_old - dk * self.û) / lhs_coef

        self.reverse[:] = self.ŷ
        self.y[:] = self.ift()

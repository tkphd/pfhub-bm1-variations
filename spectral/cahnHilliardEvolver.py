#!/usr/bin/env python3

import numpy as np
import pyfftw
import pyfftw.builders as FFTW
import pyfftw.interfaces.numpy_fft as FFT


class CahnHilliardEvolver:
    def __init__(self, y, y_old, dx, γ):
        sy = list(y.shape)
        sk = list(sy)
        sk[-1] = 1 + sk[-1] // 2

        self.dx = dx
        self.dV = dx ** len(sy)
        self.V = self.dV * np.prod(sy)

        self.A = 0.001
        self.B = 0.286
        self.γ = γ
        self.a1 = 3
        self.a2 = 0.2
        self.a1c = 1 - self.a1
        self.a2c = 1 - self.a2

        # auxiliary variables
        kx = 2 * np.pi * FFT.fftfreq(sy[0], d=self.dx)
        ky = 2 * np.pi * FFT.rfftfreq(sy[1], d=self.dx)
        k = np.array(np.meshgrid(kx, ky, indexing="ij"))

        # improved syaling in k-space, thanks to Nana Ofori-Opoku
        Kx = np.array(0.5 * (1 - np.cos(k[0])) * (3 + np.cos(k[1])))
        Ky = np.array(0.5 * (1 - np.cos(k[1])) * (3 + np.cos(k[0])))
        self.K = np.array([Kx, Ky])
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

        # compute free energy
        self.F = self.free_energy()
        self.f = self.F / self.V

    def fbulk(self):
        # bulk free energy density
        return 0.25 * (1 - self.y**2) ** 2

    def dfdc(self):
        # derivative of bulk free energy density
        # cf. TK_R6_p551
        return self.y**3 - self.y

    def dfdc_nln(self):
        # nonlinear terms
        return self.y**3

    def free_energy(self):
        fcx = FFTW.irfftn(self.ŷ * 1j * self.K[0])
        fcy = FFTW.irfftn(self.ŷ * 1j * self.K[1])

        cx = fcx().real
        cy = fcy().real

        return self.dx**2 * (0.5 * self.γ * (cx**2 + cy**2) + self.fbulk()).sum()

    def mass(self):
        return self.dV * np.sum(self.y)

    def evolve(self, dτ):
        self.y_old[:] = self.y
        self.ŷ_old[:] = self.ŷ

        self.forward[:] = self.dfdc_nln()
        self.û[:] = self.dealias * self.fft()

        dk = dτ * self.Ksq
        lin_coef = 1 + dk * (self.a1 - self.γ * self.Ksq * self.a2)
        lhs_coef = 1 - dk * (self.a1c + self.γ * self.Ksq * self.a2c)

        self.ŷ[:] = (lin_coef * self.ŷ_old - dk * self.û) / lhs_coef

        self.reverse[:] = self.ŷ
        self.y[:] = self.ift()

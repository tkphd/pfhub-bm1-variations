#!/usr/bin/env python3

import numpy as np
import pyfftw
import pyfftw.builders as FFTW
import pyfftw.interfaces.numpy_fft as FFT

π = np.pi

class UnconditionalEvolver:
    def __init__(self, c, c_old, dx, γ):
        sc = list(c.shape)
        sk = list(sc)
        sk[-1] = 1 + sk[-1] // 2

        self.dx = dx
        self.dV = dx**len(sc)
        self.V = self.dV * np.prod(sc)

        self.A  = 0.001
        self.B  = 0.286
        self.a1 = 3.0
        self.a2 = 0.2
        self.γ  = γ

        # auxiliary variables
        kx = 2.0 * π * FFT.fftfreq(sc[0], d=self.dx)
        ky = 2.0 * π * FFT.rfftfreq(sc[1], d=self.dx)
        k = np.array(np.meshgrid(kx, ky, indexing="ij"))

        # improved scaling in k-space, thanks to Nana Ofori-Opoku
        Kx = np.array((1 - np.cos(k[0])) * (3 + np.cos(k[1])) / 2)
        Ky = np.array((1 - np.cos(k[1])) * (3 + np.cos(k[0])) / 2)
        self.K = np.array([Kx, Ky])
        self.Ksq = np.sum(self.K * self.K, axis=0)

        self.nyquist = np.sqrt(self.Ksq.max()) # Nyquist mode
        self.dealias = pyfftw.byte_align(
            np.array(
                (1.5 * self.K[0] < self.nyquist) * (1.5 * self.K[1] < self.nyquist),
                dtype=bool)
        )

        # spatial arrays
        self.c       = pyfftw.zeros_aligned(sc)
        self.c_old   = pyfftw.zeros_aligned(sc)

        # FFT arrays & Plans
        self.forward = pyfftw.zeros_aligned(sc)
        self.reverse = pyfftw.zeros_aligned(sk, dtype=complex)

        self.fft = FFTW.rfftn(self.forward)
        self.ift = FFTW.irfftn(self.reverse)

        # reciprocal arrays
        self.ĉ     = pyfftw.zeros_aligned(sk, dtype=complex)
        self.ĉ_old = pyfftw.zeros_aligned(sk, dtype=complex)
        self.û     = pyfftw.zeros_aligned(sk, dtype=complex)

        # assign field values
        self.c[:]     = c
        self.c_old[:] = c_old

        self.forward[:] = self.c
        self.ĉ[:]       = self.fft()


    def fbulk(self):
        # bulk free energy density
        return 0.25 * (1 - self.c**2)**2


    def dfdc(self):
        # derivative of bulk free energy density
        # cf. TK_R6_p551
        return self.c**3 - self.c


    def dfdc_nln(self):
        # nonlinear terms
        return self.c**3


    def free_energy(self):
        fcx = FFTW.irfftn(self.ĉ * 1j * self.K[0])
        fcy = FFTW.irfftn(self.ĉ * 1j * self.K[1])

        cx = fcx().real
        cy = fcy().real

        return self.dx**2 * (self.γ/2 * (cx**2 + cy**2) + self.fbulk()).sum()


    def mass(self):
        dV = self.dx**2
        return dV * np.sum(self.c)


    def evolve(self, dt):
        self.c_old[:] = self.c
        self.ĉ_old[:] = self.ĉ

        self.forward[:] = self.dfdc_nln()
        self.û[:] = self.dealias * self.fft()

        nln_coef = dt * self.Ksq
        lin_coef = 1 + dt * self.Ksq * (self.a1 - self.γ * self.Ksq * self.a2)
        lhs_coef = 1 - dt * self.Ksq * ((1 - self.a1) + self.γ * self.Ksq * (1 - self.a2))

        self.ĉ[:] = (lin_coef * self.ĉ_old - nln_coef * self.û) / lhs_coef

        self.reverse[:] = self.ĉ
        self.c[:] = self.ift()

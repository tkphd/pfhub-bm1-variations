#!/usr/bin/env python3

import numpy as np
import os
import pyfftw
import pyfftw.builders as FFTW
import pyfftw.interfaces.numpy_fft as FFT

from .bm1 import M, κ, dfdc, dfdc_lin, dfdc_nln, fbulk


class Evolver:
    def __init__(self, c, c_old, dx):
        pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
        if "OMP_NUM_THREADS" in os.environ.keys():
            pyfftw.config.NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
        else:
            pyfftw.config.NUM_THREADS = 1

        self.dx = dx

        sc = list(c.shape)
        sk = list(sc)
        sk[-1] = 1 + sk[-1] // 2

        # auxiliary variables
        kx = 2.0 * np.pi * FFT.fftfreq(sc[0], d=self.dx)
        ky = 2.0 * np.pi * FFT.rfftfreq(sc[1], d=self.dx)
        k = np.array(np.meshgrid(kx, ky, indexing="ij"))

        # improved scaling in k-space, thanks to Nana Ofori-Opoku
        Kx = np.array((1 - np.cos(k[0])) * (3 + np.cos(k[1])) / 2)
        Ky = np.array((1 - np.cos(k[1])) * (3 + np.cos(k[0])) / 2)
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
        self.c = pyfftw.zeros_aligned(sc)
        self.c_old = pyfftw.zeros_aligned(sc)
        self.c_sweep = pyfftw.zeros_aligned(sc)

        # FFT arrays & Plans
        self.forward = pyfftw.zeros_aligned(sc)
        self.reverse = pyfftw.zeros_aligned(sk, dtype=complex)

        self.fft = FFTW.rfft2(self.forward)
        self.ift = FFTW.irfft2(self.reverse)

        # reciprocal arrays
        self.ĉ = pyfftw.zeros_aligned(sk, dtype=complex)
        self.ĉ_old = pyfftw.zeros_aligned(sk, dtype=complex)
        self.ĉ_num = pyfftw.zeros_aligned(sk, dtype=complex)  # quicker residuals
        self.û = pyfftw.zeros_aligned(sk, dtype=complex)

        # assign field values
        self.c[:] = c
        self.c_old[:] = c_old

        self.forward[:] = self.c
        self.ĉ[:] = self.fft()

        self.forward[:] = self.c_old
        self.ĉ_old[:] = self.fft()

        self.forward[:] = dfdc(self.c)
        self.û[:] = self.fft()

    def free_energy(self):
        fcx = FFTW.irfft2(self.ĉ * 1j * self.K[0])
        fcy = FFTW.irfft2(self.ĉ * 1j * self.K[1])

        cx = fcx().real
        cy = fcy().real

        return self.dx**2 * (κ / 2 * (cx**2 + cy**2) + fbulk(self.c)).sum()

    def mass(self):
        dV = self.dx**2
        return dV * np.sum(self.c)

    def residual(self, new_co):
        return np.linalg.norm(self.ĉ_num - new_co * self.ĉ).real

    def sweep(self, old_co, new_co):
        self.forward[:] = dfdc_nln(self.c_sweep, self.c)
        self.û[:] = self.dealias * self.fft()

        self.ĉ_num[:] = self.ĉ_old - old_co * self.û
        self.ĉ[:] = self.ĉ_num / new_co

        self.c_sweep[:] = self.c

        self.reverse[:] = self.ĉ
        self.c[:] = self.ift()

        return self.residual(new_co)

    def evolve(
        self, dt, maxsweeps=20, residual_tolerance=1e-12, convergence_tolerance=1e-3
    ):
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
        while (
            res > residual_tolerance or l2c > convergence_tolerance
        ) and swe < maxsweeps:
            res = self.sweep(old_co, new_co)
            l2c = np.linalg.norm(self.c - self.c_sweep)
            swe += 1

        self.c_old[:] = self.c
        self.ĉ_old[:] = self.ĉ

        if swe >= maxsweeps and res > residual_tolerance:
            raise ValueError(f"Exceeded {maxsweeps:,} sweeps with res = {res:,}")

        return swe, res, l2c


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
        value = 10 ** np.ceil(np.log10(start)).astype(int)
        delta = 10 ** np.floor(np.log10(start)).astype(int)

        while value > start:
            value -= delta
        print(f"Δ = {delta}, t = {value}")

    while True:
        value += delta
        yield value
        if value == 10 * delta:
            delta = value

#!/usr/bin/env python3

import numpy as np
import pyfftw
import pyfftw.builders as FFTW

L = 200  # domain size
M = 5.0  # diffusivity

α = 0.3  # eqm composition of phase A
β = 0.7  # eqm composition of phase B
ρ = 12.5  # well height
κ = 2.0  # gradient energy coeff

ζ = 0.5  # mean composition
ϵ = 0.01  # noise amplitude


def dfdc(c):
    # derivative of bulk free energy density
    # cf. TK_R6_p551
    return 2 * ρ * (c - α) * (β - c) * (α + β - 2 * c)


def fbulk(c):
    # bulk free energy density
    return ρ * (c - α)**2 * (β - c)**2


def free_energy(c, dx, K):
    sc = list(c.shape)
    sk = list(c.shape)
    sk[-1] //= 2

    χ = c.copy()
    ĉ = pyfftw.zeros_aligned(sk, dtype=complex)

    fft = FFTW.rfftn(χ)
    ĉ = fft()

    ĉx = ĉ * 1j * K[0]
    ĉy = ĉ * 1j * K[1]

    fcx = FFTW.irfftn(ĉx.copy())
    fcy = FFTW.irfftn(ĉy.copy())

    fgrad = pyfftw.zeros_aligned(sc)
    fgrad[:] = fcx().real**2 + fcy().real**2

    return dx**2 * (0.5 * κ * fgrad + fbulk(c)).sum()


# window function
def hann(x):
    return np.sin(np.pi * x / L) ** 2  # Hann window


def ic(x, y, variant):
    # published cosine coefficients
    A0 = np.array([0.105, 0.130, 0.025, 0.070])
    B0 = np.array([0.110, 0.087, 0.150, 0.020])

    # periodic cosine coefficients
    Ap = np.pi / L * np.array([6.0, 8.0, 2.0, 4.0])
    Bp = np.pi / L * np.array([8.0, 6.0, 10.0, 2.0])

    if variant == "original":
        values = ripples(x, y, A0, B0)
    elif variant == "periodic":
        values = ripples(x, y, Ap, Bp)
    elif variant == "window":
        values = hann(x) * hann(y) * ripples(x, y, A0, B0)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return ζ + ϵ * values


def mass(c, dV):
    return dV * np.sum(c)


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


def ripples(x, y, A, B):
    return (
        np.cos(A[0] * x) * np.cos(B[0] * y)
        + (np.cos(A[1] * x) * np.cos(B[1] * y)) ** 2
        + np.cos(A[2] * x - B[2] * y) * np.cos(A[3] * x - B[3] * y)
    )

#!/usr/bin/env python3

import numpy as np

L = 200  # domain size
M = 5.0  # diffusivity

α = 0.3  # eqm composition of phase A
β = 0.7  # eqm composition of phase B
ρ = 5.0  # well height
κ = 2.0  # gradient energy coeff

ζ = 0.5  # mean composition
ϵ = 0.01  # noise amplitude


def fbulk(c):
    # bulk free energy density
    return ρ * (c - α) ** 2 * (β - c) ** 2


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


# not-random microstructure
def ripples(x, y, A, B):
    return (
        np.cos(A[0] * x) * np.cos(B[0] * y)
        + (np.cos(A[1] * x) * np.cos(B[1] * y)) ** 2
        + np.cos(A[2] * x - B[2] * y) * np.cos(A[3] * x - B[3] * y)
    )


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

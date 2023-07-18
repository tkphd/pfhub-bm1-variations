#!/usr/bin/env python3

import numpy as np

# interfacial free energy density
finterf = lambda c_hat, 𝜅, Ksq: \
    𝜅 * np.fft.ifft2(Ksq * c_hat**2).real

# bulk free energy density f(c) = Wc²(1-c)²
fbulk = lambda c, 𝜚, 𝛼, 𝛽: \
    𝜚 * (c - 𝛼) ** 2 * (c - 𝛽) ** 2

# derivative of bulk free energy density
dfdc = lambda c, 𝜚, 𝛼, 𝛽: \
    2 * 𝜚 * (c - 𝛼) * (c - 𝛽) * (2 * c - (𝛼 + 𝛽))

c_x = lambda c_hat, K: \
    np.fft.ifft2(c_hat * 1j * K[0]).real

c_y = lambda c_hat, K: \
    np.fft.ifft2(c_hat * 1j * K[1]).real

free_energy = lambda c, c_hat, K, 𝜅, dx, 𝜚, 𝛼, 𝛽: \
    (0.5 * 𝜅 * (c_x(c_hat, K) ** 2 + c_y(c_hat, K) ** 2) + fbulk(c, 𝜚, 𝛼, 𝛽)).sum() * dx**2

def evolve_ch(c, dt, dx, M, 𝜅, 𝜚, 𝛼, 𝛽):
    N = c.shape[0]
    c_hat = np.empty((N, N), dtype=np.complex64)
    dfdc_hat = np.empty((N, N), dtype=np.complex64)
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    K = np.array(np.meshgrid(k, k, indexing="ij"), dtype=np.float32)
    Ksq = np.sum(K * K, axis=0, dtype=np.float32)

    # dealias the flux capacitor
    kmax_dealias = k.max() * 2.0 / 3.0  # The Nyquist mode
    antialias = np.array(
        (np.abs(K[0]) < kmax_dealias) * (np.abs(K[1]) < kmax_dealias), dtype=bool
    )

    c_hat[:] = np.fft.fft2(c)

    c_old = c.copy()

    dfdc_hat[:] = antialias * np.fft.fft2(dfdc(c_old, 𝜚, 𝛼, 𝛽))  # FT of the derivative

    # take a step in time
    c_hat[:] = (c_hat - dt * Ksq * M * dfdc_hat) \
             / (1 + dt * M * 𝜅 * Ksq**2)

    c_old[:] = c

    c = np.fft.ifft2(c_hat).real  # inverse fourier transform

    return c, free_energy(c, c_hat, K, 𝜅, dx, 𝜚, 𝛼, 𝛽)

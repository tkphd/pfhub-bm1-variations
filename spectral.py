#!/usr/bin/env python3

import numba
import numpy as np
import numpy.fft as FFT
import numpy.linalg as LA

π = np.pi
L = 200

α = 0.3  # eqm composition of phase A
β = 0.7  # eqm composition of phase B
ρ = 5.0  # well height
κ = 2.0  # gradient energy coeff
M = 5.0  # diffusivity

def finterf(c_hat, Ksq):
    # interfacial free energy density
    return κ * FFT.ifft2(Ksq * c_hat**2).real


def fbulk(c):
    # bulk free energy density
    return ρ * (c - α)**2 * (β - c)**2


def dfdc(c):
    # derivative of bulk free energy density
    return 2 * ρ * (c - α) * (β - c) * (α + β - 2 * c)


def dfdc_nonlinear(c):
    return 2 * ρ * (2 * c**3 - 3 * (α + β) * c**2 - α**2 * β - α * β**2)


def c_x(c_hat, K):
    return FFT.ifft2(c_hat * 1j * K[0]).real


def c_y(c_hat, K):
    return FFT.ifft2(c_hat * 1j * K[1]).real


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

        self.c_hat = FFT.fft2(self.c)
        self.c_hat_prev = np.ones_like(self.c_hat)
        self.c_hat_old = self.c_hat.copy()

        self.dfdc_hat = np.ones_like(self.c_hat)

        # prepare auxiliary arrays
        k = 2 * π * FFT.fftfreq(self.c.shape[0], d=self.dx)
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
        return LA.norm(np.abs(self.c_hat_old - numer_coeff * self.dfdc_hat
                              - denom_coeff * self.c_hat_prev).real)


    def sweep(self, numer_coeff, denom_coeff):
        self.c_hat_prev[:] = self.c_hat

        self.dfdc_hat[:] = self.alias_mask * FFT.fft2(dfdc_nonlinear(self.c_sweep))

        self.c_hat[:] = (self.c_hat_old - numer_coeff * self.dfdc_hat) / denom_coeff

        self.c[:] = FFT.ifft2(self.c_hat).real

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


# === Coincident Interpolation ===
# performant implementation after @stvdwtt

class CoincidentInterpolant:
    """
    O(h²)-accurate periodic interpolation framework
    """
    def __init__(self, shape):
        """
        Set the "fine mesh" details in real & reciprocal space
        """
        self.shape = shape
        self.compute_wave_numbers()
        self.fine_hat = np.zeros(self.shape, dtype=complex)

    def compute_wave_numbers(self):
        """
        Generate mesh points in reciprocal space
        """
        kx = np.zeros(self.shape, dtype=complex)
        ky = np.zeros(self.shape, dtype=complex)

        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                kx[i][j] = i;
                ky[i][j] = j;

                if (i > self.shape[0]/2):
                    kx[i][j] = i - self.shape[0]
                if (j > self.shape[1] / 2):
                    ky[i][j] = j - self.shape[1]


                if (2*i == self.shape[0]):
                    kx[i][j] = 0.0
                if (2*j == self.shape[1]):
                    ky[i][j] = 0.0

        self.kx = 2 * π * complex(0,1) * kx/ self.shape[0]
        self.ky = 2 * π * complex(0,1) * ky / self.shape[1]


    def upsample(self, coarse):
        """
        Interpolate coarse field data onto the fine mesh

        (needs much more explanation)
        """
        for i in range(len(self.shape)):
            if not (self.shape[i] / coarse.shape[i]).is_integer():
                raise ValueError(f"Dim. {i}: field size is mismatched!")

        scale_x = self.shape[0] // coarse.shape[0]
        scale_y = self.shape[1] // coarse.shape[1]

        coarse_hat = FFT.fft2(coarse)

        half_x = coarse_hat.shape[0] // 2
        half_y = coarse_hat.shape[1] // 2

        # Re-map values on the interval [a, b] = [-Nx//2, Nx//2+1]
        a_x =-half_x
        b_x = half_x + 1
        a_y =-half_y
        b_y = half_y + 1

        self.fine_hat[0:b_x, 0:b_y] = coarse_hat[0:b_x, 0:b_y]
        self.fine_hat[a_x :, 0:b_y] = coarse_hat[-a_x:, 0:b_y]
        self.fine_hat[0:b_x, a_y :] = coarse_hat[0:b_x, -a_y:]
        self.fine_hat[a_x :, a_y :] = coarse_hat[-a_x:, -a_y:]

        self.fine_hat[b_x,  : ] = self.fine_hat[b_x,  : ] / 2
        self.fine_hat[ : , b_y] = self.fine_hat[:  , b_y] / 2

        self.fine_hat[a_x,  : ] = self.fine_hat[a_x,  : ] / 2
        self.fine_hat[ : , a_y] = self.fine_hat[:  , a_y] / 2

        fine = FFT.ifft2(self.fine_hat) * scale_x * scale_y

        return fine.real


# === Spectral Interpolation ===
# accurate implementation after Trefethen

@numba.njit(cache=True)
def Sn(x, hc):
    # Periodic sinc function
    # Trefethen Eq. (3.7)
    return 1.0 if np.abs(x) < 1e-5 \
        else (hc * np.sin(π * x / hc)) / (2 * π * np.tan(x / 2))


@numba.njit(parallel=True)
def generate_hash_table(Nx_fine, Nx_coarse):
    """
    Create the hash table, given
    - Nf, Nc are even and domain is reasonably well refined
    - hc/hf is an integer
    """

    table = np.zeros(Nx_fine, dtype=np.float64)

    hf = 2 * π / Nx_fine
    hc = 2 * π / Nx_coarse

    N_ratio = Nx_fine // Nx_coarse

    for i in numba.prange(Nx_fine - 1):
        xf = i * hf
        for k in range(Nx_coarse):
            xc = k * hc

            idx = abs(i - N_ratio * k)
            table[idx] = Sn(xf - xc, hc)

            # If multiple dx map to the same index, that's a collision.
            # collision = (not np.isclose(table[idx], 0.0) and \
            #              not np.isclose(table[idx], tmp))

            # if not collision:
            #    table[idx] = tmp
            # else:
            #     # raise KeyError(f"Collision @ {idx}, {tmp} != {table[idx]}")
            #     table[idx] = np.nan

    return table

@numba.njit(parallel=True)
def interpolate(coarse, Nx_fine, table):
    """
    Interpolate coarse field data onto the fine mesh
    """
    Nx_coarse = coarse.shape[0]
    N_ratio = Nx_fine // Nx_coarse

    # hc = L / Nx_coarse
    # hf = L / Nx_fine

    fine = np.zeros((Nx_fine, Nx_fine), dtype=np.float64)

    # fine/outer loop
    for i in numba.prange(Nx_fine):
        for j in range(Nx_fine):

            # coarse/inner loop
            for k in range(Nx_coarse):
                idx = abs(i - N_ratio * k)
                s_x = table[idx]
                for l in range(Nx_coarse):
                    idy = abs(j - N_ratio * l)
                    s_y = table[idy]

                    fine[i, j] += s_x * s_y * coarse[k, l]

    return fine


class SpectralInterpolant:
    """
    Spectrally accurate real-space interpolation framework for periodic data
    """
    def __init__(self, shape):
        """
        Set the "fine mesh" details in real & reciprocal space
        """
        self.shape = shape
        self.hf = L / shape[0]
        self.fine = None
        self.table = None


    def upsample(self, coarse):
        self.table = generate_hash_table(self.shape[0], coarse.shape[0])
        # generate_hash_table.parallel_diagnostics(level=4)

        self.fine = interpolate(coarse, self.shape[0], self.table)
        # interpolate.parallel_diagnostics(level=4)

        return self.fine


class FourierInterpolant:
    """
    Spectrally accurate reciprocal-space interpolation framework
    """
    def __init__(self, shape):
        """
        Set the "fine mesh" details in real & reciprocal space
        """
        self.shape = shape
        self.fine = None


    def pad(self, v):
        """
        Zero-pad an $n$-dimensional array $v$ to match the fine-mesh size
        """
        zs = np.array(np.array(self.shape) - np.array(v.shape), dtype=int).T // 2
        return np.pad(v, zs)


    def upsample(self, coarse):
        v_hat = np.fft.fftshift(np.fft.fft2(coarse))
        u_hat = self.pad(v_hat)

        scale = np.prod(self.shape) / np.prod(coarse.shape)  # == (M/N)**2 for square meshes
        self.fine = scale * np.fft.ifft2(np.fft.ifftshift(u_hat)).real

        return self.fine

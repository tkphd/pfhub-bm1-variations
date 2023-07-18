#!/usr/bin/env python
# coding: utf-8

# # PFHub BM 1a in FiPy with Steppyngstounes
#
# This notebook implements variations on PFHub Benchmark 1a (Spinodal
# Decomposition) using PyCahnHilliard and steppyngstounes.
# The goal is to explore alternative initial conditions that are periodic near
# the boundaries but otherwise match the specification.

from argparse import ArgumentParser
import numpy as np
import os
from steppyngstounes import CheckpointStepper, FixedStepper
import time

from spectral import evolve_ch, free_energy

startTime = time.time()

hat = lambda x: 0.5 * (1 + np.tanh(np.pi * x / 10)) * (1 + np.tanh(np.pi * (L - x) / 10)) - 1

ic = lambda x, y, A, B: \
    ζ + ϵ * hat(x) * hat(y) * (
        np.cos(A[0] * x) * np.cos(B[0] * y)
     + (np.cos(A[1] * x) * np.cos(B[1] * y)) ** 2
      + np.cos(A[2] * x - B[2] * y) \
      * np.cos(A[3] * x - B[3] * y)
    )


def log_points(t0, t1):
    """
    Return values uniformly spaced in log₂
    """
    log_dt = np.log10(2) / 2
    log_t0 = np.log2(t0)
    log_t1 = np.log2(t1 + log_dt)
    n_pnts = np.ceil((log_t1 - log_t0) / log_dt).astype(int)
    return np.unique(np.rint(np.logspace(log_t0, log_t1, base=10., num=n_pnts)).astype(int))


def spectral_energy(c, K):
    c_hat = np.fft.fft2(c)

    return free_energy(c, c_hat, K, 𝜅, dx, 𝜚, 𝛼, 𝛽)


def start_report():
    e_file = f"{iodir}/ene.csv"
    with open(e_file, "w") as fh:
        fh.write("runtime,time,free_energy\n")


def write_and_report(t, c, energies):
    np.savez_compressed(f"{iodir}/c_{t:08.0f}.npz", c=c)

    if energies is not None:
        with open(f"{iodir}/ene.csv", "a") as fh:
            for row in energies:
                fh.write("{},{},{}\n".format(*row))


# Read command line arguments

parser = ArgumentParser()
parser.add_argument("iodir", help="root directory for output files")
parser.add_argument("dx", help="mesh spacing", type=float)
parser.add_argument("dt", help="timestep", type=float)
args = parser.parse_args()

dx = args.dx
dt = args.dt
iodir = f"{args.iodir}/dx{dx:.03f}_dt{dt:.03f}"

# System parameters & kinetic coefficients

L = 200.
N = np.rint(L / dx).astype(int)

α = 0.3   # eqm composition of phase A
β = 0.7   # eqm composition of phase B
ρ = 5.0   # well height
κ = 2.0   # gradient energy coeff
M = 5.0   # diffusivity
ζ = 0.5   # mean composition
ϵ = 0.01  # noise amplitude

if not os.path.exists(iodir):
    print("Saving output to", iodir)
    os.mkdir(iodir)

# === prepare to evolve ===

t = 0.0
energies = None

start = 1.0
stop = 2e6
stops = np.unique(log_points(start, stop))

start_report()

# === generate the initial condition ===

A = 1 / L * np.array([21.0, 26.0, 5.0, 14.0])  # [0.105, 0.130, 0.025, 0.070]
B = 1 / L * np.array([22.0, 17.4, 30.0, 4.0])  # [0.110, 0.087, 0.150, 0.020]

x = np.linspace(0., L, N)
X, Y = np.meshgrid(x, x, indexing="xy")

k = 2 * np.pi * np.fft.fftfreq(N, d=dx).real
K = np.array(np.meshgrid(k, k, indexing="ij"))

c = ic(X, Y)

energies = [[time.time() - startTime, t, spectral_energy(c, K)]]

write_and_report(t, c, energies)

for check in CheckpointStepper(start=t, stops=stops, stop=stop):
    energies = []

    for step in FixedStepper(start=check.begin, stop=check.end, size=dt):
        dt = step.size

        c, nrg = evolve_ch(c, dt, dx, M, κ, ρ, α, β)

        t+= dt

        elapsed = time.time() - startTime

        energies.append([elapsed, t, nrg])

        _ = step.succeeded()

    dt = step.want

    write_and_report(t, c, energies)

    _ = check.succeeded()

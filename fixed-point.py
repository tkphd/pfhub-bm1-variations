#!/usr/bin/env python3
# coding: utf-8

# # PFHub BM 1a in FiPy with Steppyngstounes
#
# This notebook implements variations on PFHub Benchmark 1a (Spinodal
# Decomposition) using PyCahnHilliard and steppyngstounes.
# The goal is to explore alternative initial conditions that are periodic near
# the boundaries but otherwise match the specification.

from argparse import ArgumentParser
import csv
import numpy as np
import os
from steppyngstounes import CheckpointStepper
import sys
import time

sys.path.append(os.path.dirname(__file__))

from spectral import Evolver

# Start the clock
startTime = time.time()

# System parameters & kinetic coefficients

L = 200.
π = np.pi

# Read command-line flags

parser = ArgumentParser()

parser.add_argument("variant", help="variant type",
                    choices=["original", "periodic", "window"])
parser.add_argument("-x", "--dx", help="mesh resolution", type=float)
parser.add_argument("-t", "--dt", help="time resolution", type=float)

args = parser.parse_args()
dx = args.dx
dt = args.dt

iodir = f"{args.variant}/dt{dt:6.04f}_dx{dx:08.04f}"

if not os.path.exists(iodir):
    print("Saving output to", iodir)
    os.mkdir(iodir)


def stopwatch(clock):
    return np.round(time.time() - clock, 2)


def start_report():
    e_file = f"{iodir}/ene.csv"
    header = "time,iteration,free_energy,residual"
    with open(e_file, "w") as fh:
        fh.write(f"{header}\n")


def write_checkpoint(t, evolver, energies, fname):
    np.savez_compressed(fname, c=evolver.c, c_old=evolver.c_old)

    if energies is not None:
        with open(f"{iodir}/ene.csv", "a") as fh:
            writer = csv.writer(fh)
            writer.writerows(energies)


def write_and_report(t, evolver, energies):
    write_checkpoint(t, evolver, energies, f"{iodir}/c_{t:08.0f}.npz")


# === generate the initial condition ===

ζ = 0.5    # mean composition
ϵ = 0.01   # noise amplitude
λ = L / 40 # width of periodic boundary shell

N = np.rint(L / dx).astype(int)
if N % 2 != 0:
    raise ValueError(f"N must be an even integer! Got {N} from {L}/{dx}")

x = np.linspace(0., L - dx, N)
X, Y = np.meshgrid(x, x, indexing="xy")

# published cosine coefficients
A0 = np.array([0.105, 0.130, 0.025, 0.070])
B0 = np.array([0.110, 0.087, 0.150, 0.020])

# periodic cosine coefficients
Ap = π / L * np.array([6.0, 8.0, 2.0, 4.0])
Bp = π / L * np.array([8.0, 6.0, 10., 2.0])

# not-random microstructure
ripples = lambda x, y, A, B: np.cos(A[0] * x) * np.cos(B[0] * y) \
                           +(np.cos(A[1] * x) * np.cos(B[1] * y)) ** 2 \
                           + np.cos(A[2] * x - B[2] * y) \
                           * np.cos(A[3] * x - B[3] * y)

# window function
hann = lambda x: np.sin(π * x / L)**2  # Hann window

# set IC variant
if   args.variant == "original":
    ic = lambda x, y: ζ + ϵ * ripples(x, y, A0, B0)
elif args.variant == "periodic":
    ic = lambda x, y: ζ + ϵ * ripples(x, y, Ap, Bp)
elif args.variant == "window":
    ic = lambda x, y: ζ + ϵ * hann(x) * hann(y) * ripples(x, y, A0, B0)
else:
    raise ValueError("Unknown variant {args.variant}")

# === generate or load microstructure ===

print("Launching a clean simulation")
t = 0
c = ic(X, Y)

start_report()
evolve_ch = Evolver(c, c, dx)

# === prepare to evolve ===

residual = 1e-5
energies = [[time.time() - startTime, t, evolve_ch.free_energy()]]

write_and_report(t, evolve_ch, energies)

for check in CheckpointStepper(start=t,
                               stops=range(1000),
                               stop=999):

    residual, sweeps = evolve_ch.solve(dt, sweeps=1)

    t += sweeps
    energies = [[stopwatch(startTime), t, evolve_ch.free_energy(), residual]]
    write_and_report(t, evolve_ch, energies)

    if not np.all(np.isfinite(evolve_ch.c)):
        raise ValueError("Result is not Real!")

    _ = check.succeeded()

print("\nSimulation complete.")

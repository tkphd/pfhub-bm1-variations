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
import glob
import numpy as np
import os
from parse import parse
from steppyngstounes import CheckpointStepper, FixedStepper
import time

cluster_job = ("SLURM_PROCID" in os.environ)

if not cluster_job:
    from tqdm import tqdm

from spectral import Evolver

# Start the clock
startTime = time.time()

def elapsed(clock):
    return np.round(time.time() - clock, 2)


def progression():
    """
    Generate a sequence of numbers that progress in logarithmic space:
    1, 2,.. 10, 20,.. 100, 200,.. 1000, 2000, etc.
    but *don't* store them all in memory!

    Thanks to @reid-a for contributing this generator.
    """
    delta = 1
    value = 0
    while True:
        value += delta
        yield value
        if (value == 10*delta):
            delta = value


def start_report():
    e_file = f"{iodir}/ene.csv"
    header = "runtime,time,free_energy,residual,sweeps"
    with open(e_file, "w") as fh:
        fh.write(f"{header}\n")


def write_and_report(t, c, energies):
    np.savez_compressed(f"{iodir}/c_{t:08.0f}.npz", c=c)

    if energies is not None:
        with open(f"{iodir}/ene.csv", "a") as fh:
            writer = csv.writer(fh)
            writer.writerows(energies)


# Read command-line flags

parser = ArgumentParser()

parser.add_argument("iodir", help="root directory for output files")
parser.add_argument("dx", help="mesh spacing", type=float)
parser.add_argument("dt", help="timestep", type=float)

args = parser.parse_args()
dx = args.dx
dt = args.dt

iodir = f"{args.iodir}/dt{dt:6.04f}_dx{dx:6.04f}"

if not os.path.exists(iodir):
    print("Saving output to", iodir)
    os.mkdir(iodir)

# System parameters & kinetic coefficients

t_final = 2e6
L = 200.
N = np.rint(L / dx).astype(int)

x = np.linspace(0., L, N)
X, Y = np.meshgrid(x, x, indexing="xy")

ζ = 0.5   # mean composition
ϵ = 0.01  # noise amplitude
λ = 0.04 * L  # width of periodic boundary shell

# === generate the initial condition ===

# published cosine coefficients
A0 = np.array([0.105, 0.130, 0.025, 0.070])  # 1 / L * np.array([21., 26., 5.0, 14.])
B0 = np.array([0.110, 0.087, 0.150, 0.020])  # 1 / L * np.array([22., 17., 30., 4.0])

# periodic cosine coefficients
Ap = np.pi / L * np.array([6.0, 8.0, 2.0, 4.0])
Bp = np.pi / L * np.array([8.0, 6.0, 10., 2.0])

# spherical cosine coefficients
As = np.array([8.0, 12., 2.5, 7.0])
Bs = np.array([15., 10., 1.5, 2.0])

# not-random microstructure
ripples = lambda x, y, A, B: np.cos(A[0] * x) * np.cos(B[0] * y) \
                           +(np.cos(A[1] * x) * np.cos(B[1] * y)) ** 2 \
                           + np.cos(A[2] * x - B[2] * y) \
                           * np.cos(A[3] * x - B[3] * y)

tophat = lambda x: 0.5 * (1 + np.tanh(np.pi * x / λ)) \
                       * (1 + np.tanh(np.pi * (L - x) / λ)) - 1

ic_orig = lambda x, y: ζ + ϵ * ripples(x, y, A0, B0)

ic_phat = lambda x, y: ζ + ϵ * tophat(x) * tophat(y) * ripples(x, y, A0, B0)

ic_peri = lambda x, y: ζ + ϵ * ripples(x, y, Ap, Bp)

# === generate or load microstructure ===

npz_files = sorted(glob.glob(f"{iodir}/c_*.npz"))
resuming = (len(npz_files) != 0)

if resuming:
    _, t = parse("{}/c_{}.npz", npz_files[-1])
    with open(npz_files[-1], "r") as fh:
        npz = np.load(fh)
        c = npz.c
else:
    start_report()
    t = 0.0
    c = ic_orig(X, Y)

# === prepare to evolve ===

evolve_ch = Evolver(c, dx)

if resuming:
    residual = 1.0
    energies = []
else:
    residual = 1e-5
    energies = [[time.time() - startTime, t, evolve_ch.free_energy(), residual, 0]]

    write_and_report(t, c, energies)

if not cluster_job:
    tq_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n:>7,d}/{total:<7,d} {elapsed} + {remaining}{postfix}"

for check in CheckpointStepper(start=t,
                               stops=progression(),
                               stop=t_final):
    energies = []

    if cluster_job:
        # TQDM isn't appropriate when stdout redirects to a file.
        stepper = FixedStepper(start=check.begin, stop=check.end, size=dt)
    else:
        # Progress bars are delightful for local execution.
        stepper = tqdm(FixedStepper(start=check.begin, stop=check.end, size=dt),
                    desc=f"t->{check.end:7,.0f}",
                    total=np.ceil((check.end - check.begin) / dt).astype(int),
                    bar_format=tq_fmt,
                    ncols=79)

    for step in stepper:
        dt = step.size

        energy, residual, sweeps = evolve_ch.solve(dt)

        t += dt

        energies.append([elapsed(startTime), t, energy, residual, sweeps])

        _ = step.succeeded()

    dt = step.want

    write_and_report(t, evolve_ch.c, energies)

    if not np.all(np.isfinite(evolve_ch.c)):
        raise ValueError("Result is not Real!")

    _ = check.succeeded()

#!/usr/bin/env python3
# coding: utf-8

# # PFHub BM 1a in FiPy with Steppyngstounes
#
# This program implements variations on PFHub Benchmark 1a
# (Spinodal Decomposition) using PyCahnHilliard and steppyngstounes.
# The goal is to explore alternative initial conditions that are periodic near
# the boundaries but otherwise match the specification.

from argparse import ArgumentParser
import csv
import glob
import numpy as np
import os
import pandas as pd
from steppyngstounes import CheckpointStepper, PIDStepper
import sys
import time

cluster_job = bool("SLURM_PROCID" in os.environ)

sys.path.append(os.path.dirname(__file__))

from spectral import Evolver, M, κ, progression

# Start the clock
startTime = time.time()

# System parameters & kinetic coefficients

t_final = 1_500_000
L = 200.
π = np.pi

# Read command-line flags

parser = ArgumentParser()

parser.add_argument("variant", help="variant type",
                    choices=["original", "periodic", "window"])
parser.add_argument("-x", "--dx", help="mesh resolution", type=float)

args = parser.parse_args()
dx = args.dx
dt = dx**5 / (M * κ)

iodir = f"{args.variant}/dx{dx:08.04f}"
chkpt = f"{iodir}/checkpoint.npz"

if not os.path.exists(iodir):
    print("Saving output to", iodir)
    os.mkdir(iodir)


def stopwatch(clock):
    return np.round(time.time() - clock, 4)


def start_report():
    e_file = f"{iodir}/ene.csv"
    header = "runtime,time,free_energy,residual"
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

    if os.path.exists(chkpt):
        os.remove(chkpt)


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

npz_files = sorted(glob.glob(f"{iodir}/c*.npz"))
resuming = (len(npz_files) != 0) and os.path.exists(f"{iodir}/ene.csv")

if resuming:
    ene_df = pd.read_csv(f"{iodir}/ene.csv")
    print(ene_df.tail())
    t = float(ene_df.time.iloc[-1])
    startTime -= ene_df.runtime.iloc[-1]
    last_npz = npz_files[-1]

    print(f"Resuming from {last_npz} (t={t})")

    with np.load(last_npz) as npz:
        c = npz["c"]
        c_old = npz["c_old"]

    evolve_ch = Evolver(c, c_old, dx)

else:
    print("Launching a clean simulation")
    c = ic(X, Y)
    t = 0.0
    start_report()
    evolve_ch = Evolver(c, c, dx)

# Don't resume finished jobs.
if t >= t_final or np.isclose(t, t_final):
    sys.exit()

# === prepare to evolve ===

if resuming:
    residual = 1.0
    energies = []
else:
    residual = 1e-3
    energies = [[time.time() - startTime, t, evolve_ch.free_energy(), residual]]

    write_and_report(t, evolve_ch, energies)


checkTime = time.time()

for check in CheckpointStepper(start=t,
                               stops=progression(int(t)),
                               stop=t_final):
    energies = []

    for step in PIDStepper(start=check.begin,
                           stop=check.end,
                           size=dt,
                           limiting=False,
                           proportional=0.075,  # 0.075 default
                           integral=0.175,      # 0.175
                           derivative=0.005):   # 0.100
        dt = step.size
        residual, swp = evolve_ch.solve(dt, sweeps=1)
        scaled_error = residual / 1e-3

        t += dt
        energies.append([stopwatch(startTime), t, evolve_ch.free_energy(), residual])

        if stopwatch(checkTime) > 86400:
            # Write a checkpoint at least once a day
            write_checkpoint(t, evolve_ch, energies, chkpt)
            checkTime = time.time()
            energies = []

        _ = step.succeeded(error=scaled_error)

    dt = step.want

    write_and_report(t, evolve_ch, energies)

    if not np.all(np.isfinite(evolve_ch.c)):
        raise ValueError("Result is not Real!")

    _ = check.succeeded()

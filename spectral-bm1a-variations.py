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
# from line_profiler import profile
import numpy as np
import os
import pandas as pd
try:
    from rich import print
except ImportError:
    pass
from steppyngstounes import CheckpointStepper, FixedStepper
import sys
import time

cluster_job = bool("SLURM_PROCID" in os.environ)
if not cluster_job:
    from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from spectral import Evolver, M, κ, progression

# Start the clock
startTime = time.time()

# System parameters & kinetic coefficients

t_final = 100_000
L = 200.
π = np.pi

h0 = 2**-4   # 0.0625
k0 = 2**-20  # 9.5367431640625e-07

# | k0     | CFL      |
# | ---    | ---      |
# | 2**-19 | 1.25e+00 |
# | 2**-20 | 6.25e-01 |

# Read command-line flags

parser = ArgumentParser()

parser.add_argument("variant", help="variant type",
                    choices=["noise", "original", "periodic", "window", "."])
parser.add_argument("-x", "--dx",
                    type=float,
                    default=h0,
                    help=f"mesh resolution: gold standard Δx={h0}")
parser.add_argument("-t", "--dt",
                    type=float,
                    default=k0,
                    help=f"time resolution: gold standard Δt={k0}")

args = parser.parse_args()
dx = args.dx
dt = args.dt

print(f"Linear stability ~> {κ * M * dt * dx**(-4):.2e}")

if args.variant == ".":
    variant = os.path.basename(os.path.realpath("."))
else:
    variant = args.variant

iodir = f"{args.variant}/dt{dt:8.06f}_dx{dx:08.04f}"
chkpt = f"{iodir}/checkpoint.npz"

if not os.path.exists(iodir):
    print("Saving output to", iodir)
    os.mkdir(iodir)


def stopwatch(clock):
    return np.round(time.time() - clock, 2)


def start_report():
    e_file = f"{iodir}/ene.csv"
    e_head = "runtime,time,free_energy"
    with open(e_file, "w") as fh:
        fh.write(f"{e_head}\n")

    r_file = f"{iodir}/res.csv"
    r_head = "time,sweeps,residual"
    with open(r_file, "w") as fh:
        fh.write(f"{r_head}\n")


# @profile
def report(fname, lines):
    if lines is not None and len(lines) != 0:
        with open(fname, "a") as fh:
            writer = csv.writer(fh)
            writer.writerows(lines)


# @profile
def write_checkpoint(t, evolver, energies, fname):
    np.savez_compressed(fname,
                        t=t,
                        c=evolver.c,
                        c_old=evolver.c_old)

    report(f"{iodir}/ene.csv", energies)


# @profile
def write_and_report(t, evolver, energies):
    write_checkpoint(t, evolver, energies, f"{iodir}/c_{t:08.0f}.npz")

    if os.path.exists(chkpt):
        os.remove(chkpt)


# === generate the initial condition ===

ζ = 0.5    # mean composition
ϵ = 0.01   # noise amplitude

N = np.rint(L / dx).astype(int)
if N % 2 != 0:
    raise ValueError(f"N must be an even integer! Got {N} from {L}/{dx}")

x = np.linspace(0., L - dx, N)
X, Y = np.meshgrid(x, x, indexing="xy")

# not-random microstructure
# @profile
def ripples(x, y, A, B):
    return np.cos(A[0] * x) * np.cos(B[0] * y) \
         +(np.cos(A[1] * x) * np.cos(B[1] * y)) ** 2 \
         + np.cos(A[2] * x - B[2] * y) \
         * np.cos(A[3] * x - B[3] * y)

# window function
# @profile
def hann(x):
    return np.sin(π * x / L)**2  # Hann window

# @profile
def ic(x, y):
    # published cosine coefficients
    A0 = np.array([0.105, 0.130, 0.025, 0.070])
    B0 = np.array([0.110, 0.087, 0.150, 0.020])

    # periodic cosine coefficients
    Ap = π / L * np.array([6.0, 8.0, 2.0, 4.0])
    Bp = π / L * np.array([8.0, 6.0, 10., 2.0])

    coeff = ϵ

    if variant == "noise":
        prng = np.random.default_rng()  # PCG64
        values = 2 * prng.random((N, N)) - 1
    elif variant == "original":
        values = ripples(x, y, A0, B0)
    elif variant == "periodic":
        values = ripples(x, y, Ap, Bp)
    elif variant == "window":
        coeff = ϵ * hann(x) * hann(y)
        values = ripples(x, y, A0, B0)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return ζ + coeff * values

# @profile
def main():
    global dt, startTime

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
        print(f"Launching a clean '{variant}' simulation")
        c = ic(X, Y)
        t = 0.0
        start_report()
        evolve_ch = Evolver(c, c, dx)

    # Don't resume finished jobs.
    if t >= t_final or np.isclose(t, t_final):
        sys.exit()

    # === prepare to evolve ===

    if not resuming:
        energies = [[time.time() - startTime, t, evolve_ch.free_energy()]]
        residues = [[t, 0, 1e-4]]
        write_and_report(t, evolve_ch, energies)
        report(f"{iodir}/res.csv", residues)

    checkTime = time.time()

    for check in CheckpointStepper(start=t,
                                   stops=progression(int(t)),
                                   stop=t_final):
        energies = []
        residues = []
        stepper = FixedStepper(start=check.begin, stop=check.end, size=dt)

        if not cluster_job:
            # TQDM isn't appropriate when stdout redirects to a file.
            stepper = tqdm(stepper,
                           desc=f"t->{check.end:7,.0f}",
                           total=int((check.end - check.begin) / dt),
                           ncols=79)

        for step in stepper:
            dt = step.size
            residual, sweeps = evolve_ch.solve(dt)
            t += dt

            residues.append([t, sweeps, residual])

            if not np.isclose(dt, step.want) or stopwatch(checkTime) > 600:  # last step or every 10 minutes
                energies.append([stopwatch(startTime), t, evolve_ch.free_energy()])
                report(f"{iodir}/ene.csv", energies)
                report(f"{iodir}/res.csv", residues)
                energies = []
                residues = []
                checkTime = time.time()

            _ = step.succeeded()

        dt = step.want

        write_and_report(t, evolve_ch, energies)
        report(f"{iodir}/res.csv", residues)

        if not np.all(np.isfinite(evolve_ch.c)):
            raise ValueError("Result is not Real!")

        _ = check.succeeded()

if __name__ == "__main__":
    main()

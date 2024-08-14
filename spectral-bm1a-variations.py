#!/usr/bin/env python3
# coding: utf-8

# This notebook implements variations on PFHub Benchmark 1a
# (Spinodal Decomposition) using pyfftw and steppyngstounes.
# The goal is to explore initial conditions that are periodic
# near the boundaries but otherwise match the specification.

from argparse import ArgumentParser
import csv
import gzip
import glob
import numpy as np
import os
import pandas as pd
try:
    from rich import print
except ImportError:
    pass
from steppyngstounes import CheckpointStepper, PIDStepper
import sys
import time

sys.path.append(os.path.dirname(__file__))
from spectral import Evolver, L, M, κ, progression

# Start the clock
startTime = time.time()

# System parameters & kinetic coefficients

t_final = 2_000_000
π = np.pi

h0 = 2**-4   # 0.0625
k0 = 2**-20  # 9.5367431640625e-07

res_tol = 1e-7  # sweep: tolerance for residual
con_tol = 0.01  # sweep: composition convergence
max_its = 1000  # maximum sweeps to achieve tolerance

# Read command-line flags

parser = ArgumentParser()

parser.add_argument("variant", help="variant type",
                    choices=["noise", "original", "periodic", "window", "."])
parser.add_argument("-x", "--dx",
                    type=float,
                    default=h0,
                    help=f"mesh resolution: gold standard Δx={h0}")

args = parser.parse_args()
dx = args.dx
dt = dx**4 / (8 * κ * M)
# stab = κ * M * dt / dx**4

print(f"Δt ~> {dt}")

if args.variant == ".":
    variant = os.path.basename(os.path.realpath("."))
else:
    variant = args.variant

iodir = f"{args.variant}/dx{dx:08.04f}"
chkpt = f"{iodir}/checkpoint.npz"

ene_file = f"{iodir}/ene.csv.gz"

if not os.path.exists(iodir):
    os.mkdir(iodir)
print("Saving output to", iodir)


def stopwatch(clock):
    return np.round(time.time() - clock, 2)


def start_report():
    e_head = "runtime,time,free_energy,mass,dt,its,res,l2c"
    with gzip.open(ene_file, "wt") as fh:
        fh.write(f"{e_head}\n")


def report(fname, lines):
    if lines is not None and len(lines) != 0:
        with gzip.open(fname, "at") as fh:
            writer = csv.writer(fh)
            writer.writerows(lines)


def write_checkpoint(t, evolver, energies, fname):
    np.savez_compressed(fname,
                        t=t,
                        c=evolver.c,
                        c_old=evolver.c_old)

    report(ene_file, energies)


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
def ripples(x, y, A, B):
    return np.cos(A[0] * x) * np.cos(B[0] * y) \
         +(np.cos(A[1] * x) * np.cos(B[1] * y)) ** 2 \
         + np.cos(A[2] * x - B[2] * y) \
         * np.cos(A[3] * x - B[3] * y)

# window function
def hann(x):
    return np.sin(π * x / L)**2  # Hann window

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


def main():
    global dt, startTime

    # === generate or load microstructure ===

    npz_files = sorted(glob.glob(f"{iodir}/c*.npz"))
    resuming = (len(npz_files) != 0) and os.path.exists(ene_file)

    if resuming:
        ene_df = pd.read_csv(ene_file)
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
        energies = [[stopwatch(startTime), t, evolve_ch.free_energy(), evolve_ch.mass(), dt, max_its, res_tol, con_tol]]
        write_and_report(t, evolve_ch, energies)


    for check in CheckpointStepper(start=t,
                                   stops=progression(int(t)),
                                   stop=t_final):
        energies = []

        for step in PIDStepper(start=check.begin,
                               stop=check.end,
                               size=dt,
                               limiting=False,
                               proportional=0.080,
                               integral=0.175,
                               derivative=0.005):
            dt = step.size
            sweeps, residual, convergence = evolve_ch.evolve(dt, max_its, res_tol, con_tol)

            # weight the number of sweeps as well as the convergence
            con_err = convergence / con_tol
            swp_err = max(1.0, float(sweeps) / 9)
            rel_err = np.sqrt(con_err * swp_err)

            if not step.succeeded(error=rel_err):
                raise(ValueError(f"Could not converge with dt={dt}."))

            t += dt
            energies.append([stopwatch(startTime), t, evolve_ch.free_energy(), evolve_ch.mass(), dt, sweeps, residual, convergence])

        dt = step.want

        write_and_report(t, evolve_ch, energies)

        _ = check.succeeded()

    print(f"Simulation complete at t={t:,}.\n")

if __name__ == "__main__":
    main()

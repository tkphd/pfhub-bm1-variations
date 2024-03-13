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

t_final = 1  # 10_000  # 1_000_000
L = 200.
π = np.pi

h0 = 0.0625
k0 = 0.00006103515625

# Read command-line flags

parser = ArgumentParser()

parser.add_argument("variant", help="variant type",
                    choices=["noise", "original", "periodic", "window"])
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
    r_head = "time,sweeps,residual\n"
    with open(r_file, "w") as fh:
        fh.write(f"{r_head}\n")


def report(fname, lines):
    if lines is not None and len(lines) != 0:
        with open(fname, "a") as fh:
            writer = csv.writer(fh)
            writer.writerows(lines)


def write_checkpoint(t, evolver, energies, fname):
    np.savez_compressed(fname,
                        t=t,
                        c=evolver.c,
                        c_old=evolver.c_old)

    report(f"{iodir}/ene.csv", energies)


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

# window functions
tophat = lambda x: 0.25 * (1 + np.tanh(π * (x - λ) / λ)) \
                        * (1 + np.tanh(π * (L - x - λ) / λ))

hann = lambda x: np.sin(π * x / L)**2  # Hann window
hanc = lambda x: np.cos(π * x / L)**2  # complementary Hann window

# set IC variant
if args.variant == "noise":
    ic = None
    prng = np.random.default_rng()  # PCG64
elif args.variant == "original":
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
    if args.variant == "noise":
        c = ζ + ϵ * (2 * prng.random((N, N)) - 1)
    else:
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
    write_and_report(t, evolve_ch, energies)

    cfl = κ * M * dt * dx**(-4)
    print(f"Linear stability is {cfl:.2e}")

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
            checkTime = time.time()

        _ = step.succeeded()

    dt = step.want

    write_and_report(t, evolve_ch, energies)
    report(f"{iodir}/res.csv", residues)

    if not np.all(np.isfinite(evolve_ch.c)):
        raise ValueError("Result is not Real!")

    _ = check.succeeded()

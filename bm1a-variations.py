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
import pyfftw.config

try:
    from rich import print
except ImportError:
    pass
from steppyngstounes import CheckpointStepper
import sys
import time
from tqdm import tqdm

# import from `spectral/` in same folder as the script
sys.path.append(os.path.dirname(__file__))

from spectral.bm1 import L, ic
from spectral.conversions import free_energy, gamma, c2y, τ2t, t2τ, y2c
from spectral.evolver import progression
from spectral.cahnHilliardEvolver import CahnHilliardEvolver
from spectral.powerLawStepper import PowerLawStepper

# threaded FFTW shenanigans
nthr = float(os.environ["OMP_NUM_THREADS"])
if nthr < 1:
    raise ValueError("Why so few threads? ({nthr})")
else:
    pyfftw.config.NUM_THREADS = nthr

# System parameters & kinetic coefficients

t_final = 2_000_000

h0 = 2**-4  # 0.0625
k0 = 2**-20  # 9.5367431640625e-07

# Read command-line flags

parser = ArgumentParser()

parser.add_argument(
    "variant",
    help="variant type",
    choices=["original", "periodic", "window", "."],
)
parser.add_argument(
    "-x", "--dx", type=float, default=h0, help=f"mesh resolution: gold standard Δx={h0}"
)

args = parser.parse_args()
dx = args.dx

if args.variant == ".":
    variant = os.path.basename(os.path.realpath("."))
else:
    variant = args.variant

iodir = f"{args.variant}/dx{dx:08.04f}"

ene_file = f"{iodir}/ene.csv.gz"

if not os.path.exists(iodir):
    os.mkdir(iodir)
print("Saving output to", iodir)


def stopwatch(clock):
    return np.round(time.time() - clock, 2)


def start_report():
    e_head = "runtime,time,dτ,free_energy"
    with gzip.open(ene_file, "wt") as fh:
        fh.write(f"{e_head}\n")


def report(fname, lines):
    if lines is not None and len(lines) != 0:
        with gzip.open(fname, "at") as fh:
            writer = csv.writer(fh)
            writer.writerows(lines)


def write_checkpoint(t, c, c_old, fname):
    np.savez_compressed(fname, t=t, c=c, c_old=c_old)


def write_and_report(t, c, c_old, energies):
    write_checkpoint(t, c, c_old, f"{iodir}/c_{t:08.0f}.npz")
    report(ene_file, energies)


def main():
    # Start the clock
    startTime = time.time()

    t = τ = 0.0

    npz_files = sorted(glob.glob(f"{iodir}/c*.npz"))
    resuming = (len(npz_files) != 0) and os.path.exists(ene_file)

    if not resuming:
        # === generate initial condition ===
        print(f"Launching a clean '{variant}' simulation")

        N = np.rint(L / dx).astype(int)
        if N % 2 != 0:
            raise ValueError(f"N must be an even integer! Got {N} from {L}/{dx}")

        x = np.linspace(0.0, L - dx, N)
        X, Y = np.meshgrid(x, x, indexing="xy")

        c = ic(X, Y, variant)
        c_old = c
        start_report()

    else:
        # === load microstructure ===
        ene_df = pd.read_csv(ene_file)
        t = float(ene_df.time.iloc[-1])
        τ = t2τ(t)
        startTime -= ene_df.runtime.iloc[-1]
        last_npz = npz_files[-1]
        print(f"Resuming from {last_npz} (t={t})")

        with np.load(last_npz) as npz:
            c = npz["c"]
            c_old = npz["c_old"]

    # Don't resume finished jobs.
    if t >= t_final or np.isclose(t, t_final):
        sys.exit()

    # === prepare to evolve ===
    y = c2y(c)
    y_old = c2y(c_old)
    γ = gamma()

    evolve_ch = CahnHilliardEvolver(y, y_old, dx, γ)

    if not resuming:
        energies = [
            [stopwatch(startTime), t, t2τ(k0), free_energy(c, evolve_ch.dx, evolve_ch.K)]
        ]
        write_and_report(t, c, c_old, energies)

    for check in CheckpointStepper(start=t, stops=progression(int(t)), stop=t_final):
        τ0 = t2τ(check.begin)
        τ1 = t2τ(check.end)
        stepper = PowerLawStepper(start=τ0, stop=τ1)

        energies = []
        dτ = max(t2τ(k0), 0.001 * τ0**(2/3))

        pbar = tqdm(stepper,
                    desc=f"t->{check.end:7,.0f}",
                    total=int((τ1 - τ0) / dτ))

        for step in pbar:
            dτ = step.size
            pbar.total = int((τ1 - τ0) / dτ)
            pbar.refresh()

            evolve_ch.evolve(dτ)

            τ += dτ
            t = τ2t(τ)

            c = y2c(evolve_ch.y)
            c_old = y2c(evolve_ch.y_old)

            energies.append(
                [stopwatch(startTime), t, dτ, free_energy(c, evolve_ch.dx, evolve_ch.K)]
            )

            report(ene_file, energies)
            energies.clear()

            _ = step.succeeded(value=τ)

        write_and_report(t, c, c_old, energies)
        energies.clear()
        _ = check.succeeded()

    print(f"Simulation complete at t={t:,}.\n")


if __name__ == "__main__":
    main()

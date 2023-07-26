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
from steppyngstounes import CheckpointStepper, FixedStepper
import time
from tqdm import tqdm

from spectral import Evolver

startTime = time.time()

# Read command line arguments

parser = ArgumentParser()
parser.add_argument("iodir", help="root directory for output files")
parser.add_argument("dx", help="mesh spacing", type=float)
parser.add_argument("dt", help="timestep", type=float)
parser.add_argument("sweeps", help="number of non-linear sweeps per solve", type=int)
args = parser.parse_args()

dx = args.dx
dt = args.dt
iodir = f"{args.iodir}/dx{dx:4.02f}_dt{dt:4.02f}_sw{args.sweeps:02d}"

if not os.path.exists(iodir):
    print("Saving output to", iodir)
    os.mkdir(iodir)

# System parameters & kinetic coefficients

t_final = 50e3
L = 200.
N = np.rint(L / dx).astype(int)

ζ = 0.5   # mean composition
ϵ = 0.01  # noise amplitude
λ = 0.04 * L  # width of periodic boundary shell

A = np.array([0.105, 0.130, 0.025, 0.070])  # 1 / L * np.array([21.0, 26.0, 5.0, 14.0])
B = np.array([0.110, 0.087, 0.150, 0.020])  # 1 / L * np.array([22.0, 17.4, 30.0, 4.0])

Ap = np.pi / L * np.array([6.0, 8.0, 2.0, 4.0])
Bp = np.pi / L * np.array([8.0, 6.0,-10.,-2.0])

# smooth top-hat function
hat = lambda x: 0.5 * (1 + np.tanh(np.pi * x / λ)) * (1 + np.tanh(np.pi * (L - x) / λ)) - 1

ic = lambda x, y: \
    ζ + ϵ * hat(x) * hat(y) * (
        np.cos(A[0] * x) * np.cos(B[0] * y)
     + (np.cos(A[1] * x) * np.cos(B[1] * y)) ** 2
      + np.cos(A[2] * x - B[2] * y) \
      * np.cos(A[3] * x - B[3] * y)
    )


icp = lambda x, y: \
    ζ + ϵ * (
        np.cos(Ap[0] * x) * np.cos(Bp[0] * y)
     + (np.cos(Ap[1] * x) * np.cos(Bp[1] * y)) ** 2
      + np.cos(Ap[2] * x - Bp[2] * y) \
      * np.cos(Ap[3] * x - Bp[3] * y)
    )


def progression():
    """
    Generate a sequence of numbers that progress in logarithmic space:
    1, 2,.. 10, 20,.. 100, 200,.. 1000, 2000, etc.
    but *don't* store them all in memory!
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
    header = "runtime,time,free_energy"
    for sweep in range(args.sweeps):
        header += f",res{sweep:02}"
    with open(e_file, "w") as fh:
        fh.write(f"{header}\n")


def write_and_report(t, c, energies):
    np.savez_compressed(f"{iodir}/c_{t:08.0f}.npz", c=c)

    if energies is not None:
        with open(f"{iodir}/ene.csv", "a") as fh:
            writer = csv.writer(fh)
            writer.writerows(energies)

# === prepare to evolve ===

t = 0.0
energies = None

start_report()

# === generate the initial condition ===

x = np.linspace(0., L, N)
X, Y = np.meshgrid(x, x, indexing="xy")

c = icp(X, Y)

evolve_ch = Evolver(c, dx, args.sweeps)

# write initial energy

res = np.zeros(args.sweeps)
energies = [[time.time() - startTime, t, evolve_ch.free_energy(), *res]]

write_and_report(t, c, energies)

tq_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n:>7,d}/{total:<7,d} {elapsed} + {remaining}{postfix}"

for check in CheckpointStepper(start=t,
                               stops=progression(),
                               stop=t_final):
    energies = []

    pbar = tqdm(FixedStepper(start=check.begin, stop=check.end, size=dt),
                desc=f"t->{check.end:7,.0f}",
                total=np.ceil((check.end - check.begin) / dt).astype(int),
                bar_format=tq_fmt,
                ncols=79)

    for step in pbar:
        dt = step.size

        nrg, res = evolve_ch.solve(dt)

        t += dt

        elapsed = time.time() - startTime

        energies.append([elapsed, t, nrg, *res])

        _ = step.succeeded()

    dt = step.want

    write_and_report(t, evolve_ch.c, energies)

    _ = check.succeeded()

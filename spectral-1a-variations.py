#!/usr/bin/env python3
# coding: utf-8

# PyMKS solves:
#
# # $$ f_0 = \frac{1}{2} \left(1 - \phi \right)^2 \left(1 + \phi \right)^2 $$
#
# # $$ \dot{\phi} = \nabla^2 \left(\phi^3 - \phi \right) - \gamma \nabla^4 \phi$$
#
# `solve_cahn_hilliard` steps the above equation, spanning c = [-1, 1).
# We need to map from one composition and bulk free energy to the other:

# Problem 1(a) specified by
#
# # $$ f_a = 5 \left( 0.3 - \phi \right)^2 \left(0.7 - \phi \right)^2 $$


from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=UserWarning)

import gc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil

from argparse import ArgumentParser
from pymks import solve_cahn_hilliard
from pymks.fmks.func import curry
from steppyngstounes import CheckpointStepper
from tqdm import tqdm

mpl.use("agg")
proc = psutil.Process()

parser = ArgumentParser(
    prog = 'fipy-bm1-variations',
    description = 'PFHub BM1 in PyMKS with periodic initial condition variations'
)

parser.add_argument("variant", help="one of 'orig', 'peri', or 'zany'")

args = parser.parse_args()
iodir = args.variant

# Domain & numerical parameters
dt = 1.0  # 2**(-2)
dx = 1.0  # 2**(-2)
Lx = Ly = 200
Nx = int(Lx/dx)

cos  = np.cos
pi   = np.pi
log10= np.log10

# Define initial conditions

if args.variant == "orig":
    # BM 1a specification: not periodic at all
    A0 = np.array([0.105, 0.130, 0.025, 0.070])
    B0 = np.array([0.110, 0.087,-0.150,-0.020])
elif args.variant == "peri":
    # Even integers as close to spec as achievable:
    # exactly periodic at the domain boundaries
    A0 = pi/Lx * np.array([6.0, 8.0, 2.0, 4.0])
    B0 = pi/Ly * np.array([8.0, 6.0,-10.,-2.0])
elif args.variant == "zany":
    # Perturbation of the periodic coefficients:
    # almost periodic, visually similar to the original
    A0 = pi/Lx * np.array([6.125, 7.875, 2.125, 4.125])
    B0 = pi/Ly * np.array([7.875, 5.125,-9.875,-1.875])
else:
    raise ValueError("Variant {} undefined.".format(args.variant))

xx = np.linspace(dx / 2, Lx - dx / 2, Nx)
x, y = np.meshgrid(xx, xx)

# PyMKS coefficients
α0 = -1.0  # α-phase composition
β0 =  1.0  # β-phase composition
ζ0 = (α0 + β0) / 2  # mean composition
Δ0 = β0 - α0  # eqm. composition range
ρ0 = 0.5

# PFHub coefficients
α = 0.3  # α-phase composition
β = 0.7  # β-phase composition
ζ = (α + β) / 2  # mean composition
Δ = β - α  # eqm. composition range
ρ = 5
κ = 2
M = 5
ϵ = 0.01 # noise amplitude

# scale from PFHub space into PyMKS space (unity)
f0 = lambda c: ρ0 * (c - α0)**2 * (c - β0)**2
fa = lambda c: ρ  * (c - α)**2  * (c - β)**2

gamma = Δ**2 * (f0(ζ0) / fa(ζ)) / κ

t_fin = 5_000_000


def initialize(A, B, x, y):
    return ζ + ϵ * (
           cos(A[0] * x) * cos(B[0] * y) \
        + (cos(A[1] * x) * cos(B[1] * y))**2 \
        +  cos(A[2] * x  +     B[2] * y) \
        *  cos(A[3] * x  +     B[3] * y)
    )


initial_phi_ = lambda x, y: initialize(A0, B0, x, y)
initial_phi  = lambda x, y: (initial_phi_(x, y) - ζ) * κ / Δ

solve_ = curry(solve_cahn_hilliard)
solve = solve_(n_steps=1, delta_x=dx, delta_t=dt, gamma=gamma)

# Functions to run and calculate the free energy

# Write to disk uniformly in logarithmic space
checkpoints = np.unique(
    [
        int(float(10**q)) for q in
        np.arange(0, log10(t_fin), 0.1)
    ]
)

def calc_grad_mag(data):
    datax = np.concatenate((data[-1:, :], data, data[:1, :]), axis=0)
    datay = np.concatenate((data[:, -1:], data, data[:, :1]), axis=1)
    grad_x = 0.5 * (datax[2:, :] - datax[:-2, :]) / dx
    grad_y = 0.5 * (datay[:, 2:] - datay[:, :-2]) / dx
    return np.sum((grad_x**2 + grad_y**2))

def calc_f_total(data, dx, gamma):
    return fa(ζ) / f0(ζ0) * (
        np.sum(f0(data)) + 0.5 * gamma * calc_grad_mag(data)) * dx**2

def write_plot(data, t=0.0):
    imgname = "pymks/%s/spectral.%08d.png" % (iodir, int(t))
    if not os.path.exists(imgname):
        plt.figure()
        plt.title(r"$t = %s$" % "{:,}".format(int(t)))
        plt.imshow(data)
        plt.colorbar()
        # locs = np.linspace(0, Nx, num=5, endpoint=True, dtype=int)
        # labs = np.linspace(0, Lx, num=5, endpoint=True, dtype=int)
        # plt.xticks(locs, labs)
        # plt.yticks(locs, np.flip(labs))
        plt.savefig(imgname, bbox_inches="tight", dpi=400)
        plt.close()

def run():
    df = None
    t0 = 0
    field = np.reshape(initial_phi(x, y), (1, Nx, Nx))
    write_plot(field[0], 0)

    progress = tqdm(CheckpointStepper(start=0,
                                      stops=checkpoints,
                                      stop=t_fin),
                    unit="step",
                    total=len(checkpoints),
                    bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}')

    for check in progress:
        steps = np.arange(check.begin, check.end, dt)

        data_t = [0.0] * len(steps)
        data_f = [0.0] * len(steps)
        data_m = [0.0] * len(steps)

        for i, step in enumerate(steps):
            label = "[{:>13.2f} .. {:>13.2f})".format(step, check.end)
            progress.set_description(label)

            field = solve(field).compute()

            data_t[i] = t0 + dt * step / M
            data_f[i] = calc_f_total(field[0], dx, gamma)
            data_m[i] = proc.memory_info().rss / 1024

        row = pd.DataFrame(list(zip(data_t, data_f, data_m)),
                           columns=["time", "free_energy", "mem_KB"])

        df = pd.concat([df, row])

        _ = check.succeeded()

        t0 = df.time.iloc[-1]

        write_plot(field[0], t0)

        gc.collect()

        # Get the data and write to CSV
        df.to_csv("pymks/{}/free_energy.csv.gz".format(iodir),
                  compression="gzip", index=False)

run()

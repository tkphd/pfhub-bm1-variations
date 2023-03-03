#!/usr/bin/python3
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # PFHub BM 1a in PyMKS for Spectral Solution
#
# Based on[@wd15](https://github.com/wd15)'s work,
# <https://gist.github.com/wd15/ce3c8620fd19ba58f80e4f35da562dda#file-update-ipynb>
#
# ## Equations for PyMKS
#
# PyMKS assumes the field variable $\varphi$ has minima at -1 and +1.
#
# Free energy density:
#
# $$ f = \frac{1}{4} \left( 1 - \varphi \right)^2 \left(1 +\varphi
# \right)^2 $$
#
# The equation of motion:
#
# $$
#   \frac{\partial \varphi}{\partial \tau} = \nabla^2 \left[ \frac{\partial f}{\partial \varphi} \right] - \gamma \nabla^4 \varphi
# $$
#
# becomes
#
# $$
#   \frac{\partial \varphi}{\partial \tau} = \nabla^2 \left( \varphi^3 - \varphi \right) - \gamma \nabla^4 \varphi
# $$
#
# ## Equations for PFHub
#
# PFHub BM 1 assumes the field variable $c$ has minima at 0.3 and 0.7.
#
# The equation of motion
#
# $$
#   \frac{\partial c}{\partial t} = \nabla \cdot \left\{ M \nabla \left(  \frac{\partial f_{\text{chem}}}{\partial c} - \kappa\nabla^2 c \right) \right\}
# $$
#
# with
#
# $$
#   f_{\text{chem}} = \rho_s \left(c - c_{\alpha} \right)^2 \left(c - c_{\beta} \right)^2
# $$
#
# ## How to transform between the equations
#
# Use the following transformations:
#
# $$
#   c = \frac{1}{2}\left(c_{\beta} - c_{\alpha} \right) \left( 1 + \varphi \right) + c_{\alpha}
# $$
#
# and
#
# $$
#   t = \tilde{t} \tau
# $$
#
# Note that $\tau$ is the new time scale and $\tilde{t}$ is the transformation constant.
# With this transformation,
#
# $$
#   \frac{\partial f_{\text{chem}}}{\partial c} = \frac{\rho_s}{2} \left( c_{\beta} - c_{\alpha} \right)^3 \left( \varphi^3 - \varphi \right)
# $$
#
# after substitution. The benchmark equation then becomes
#
# $$
#   \frac{1}{\tilde{t} M \rho_s \left( c_{\beta} - c_{\alpha} \right)^2 } \dot{\varphi} = \nabla^2 \left( \varphi^3 - \varphi \right)  - \frac{\kappa}{\left( c_{\beta} - c_{\alpha} \right)^2 \rho_s } \nabla^4 \varphi
# $$
#
# So to make this work, we need to choose
#
# $$
#   \tilde{t} = \frac{1}{M \rho_s \left( c_{\beta} - c_{\alpha} \right)^2 }
# $$
#
# That's our time scale transformation between pymks and the benchmark problem. Also $\gamma$ is chosen to be
#
# $$
#   \gamma = \frac{\kappa}{\left( c_{\beta} - c_{\alpha} \right)^2 \rho_s }
# $$
#
# and
#
# $$
#   \dot{\varphi} = \frac{\partial \varphi}{\partial \tau} = \nabla^2 \left( \varphi^3 - \varphi \right) - \gamma \nabla^4 \varphi
# $$

# +
from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=UserWarning)

import gc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import time

from pymks import solve_cahn_hilliard
from pymks.fmks.func import curry
from steppyngstounes import CheckpointStepper, FixedStepper
from tqdm import tqdm
# -

# Choose your variant and numerical limits of interest

variant = "orig"  # one of "orig", "peri", or "zany"
dx = 0.9
dt = 0.1
t_fin = 200_000

startTime = time.time()

mpl.use("agg")
proc = psutil.Process()

if not os.path.exists(f"pymks/{variant}"):
    if not os.path.exists("pymks"):
        os.mkdir("pymks")
    os.mkdir(f"pymks/{variant}")

# Domain & numerical parameters

Lx = Ly = 200
Nx = Ny = int(np.ceil(Lx / dx))

# Define initial conditionsc

if variant == "orig":
    # BM 1a specification: not periodic at all
    A0 = np.array([0.105, 0.130, 0.025, 0.070])
    B0 = np.array([0.110, 0.087,-0.150,-0.020])
elif variant == "peri":
    # Even integers as close to spec as achievable:
    # exactly periodic at the domain boundaries
    A0 = np.pi/Lx * np.array([6.0, 8.0, 2.0, 4.0])
    B0 = np.pi/Ly * np.array([8.0, 6.0,-10.,-2.0])
elif variant == "zany":
    # Perturbation of the periodic coefficients:
    # almost periodic, visually similar to the original
    A0 = np.pi/Lx * np.array([6.125, 7.875, 2.125, 4.125])
    B0 = np.pi/Ly * np.array([7.875, 5.125,-9.875,-1.875])
else:
    raise ValueError("Variant {} undefined.".format(variant))

α = 0.3
β = 0.7
M = 5
κ = 2
ρ = 5
γ = κ / (ρ * (β - α)**2)

solve_ = curry(solve_cahn_hilliard)
solve = solve_(n_steps=1, delta_x=dx, gamma=γ)

# Define helper functions

def f_chem(c):
    return ρ * (c - α)**2 * (c - β)**2

def get_conc_time(time):
    return time / (M * ρ * (β - α)**2)

def get_phi_time(bench_t):
    return bench_t / get_conc_time(1.)

def get_conc(φ):
    return (1 + φ) * (β - α) / 2 + α

def get_phi(c):
    return 2 * (c - α) / (β - α) - 1

def initialize(A, B, x, y):
    return 0.5 + 0.01 * (
           np.cos(A[0] * x) * np.cos(B[0] * y) \
        + (np.cos(A[1] * x) * np.cos(B[1] * y))**2 \
        +  np.cos(A[2] * x  +        B[2] * y) \
        *  np.cos(A[3] * x  +        B[3] * y))

def calc_grad_mag_sq(conc):
    cx = np.concatenate((conc[-1:, :], conc, conc[:1, :]), axis=0)
    cy = np.concatenate((conc[:, -1:], conc, conc[:, :1]), axis=1)
    c_x = (cx[2:, :] - cx[:-2, :]) / (2 * dx)
    c_y = (cy[:, 2:] - cy[:, :-2]) / (2 * dx)

    return np.sum((c_x**2 + c_y**2))

def calc_f_total(φ, dx, κ):
    conc = get_conc(φ)
    return (np.sum(f_chem(conc)) + 0.5 * κ * calc_grad_mag_sq(conc)) * dx**2

def write_plot(data, t=0.0):
    imgname = "pymks/%s/spectral.%08d.png" % (variant, int(t))
    if not os.path.exists(imgname):
        plt.figure()
        plt.title(r"$t = %s$" % "{:,}".format(int(t)))
        plt.imshow(data)
        plt.colorbar()
        plt.savefig(imgname, bbox_inches="tight", dpi=400)
        plt.close()


initial_conc = lambda x, y: initialize(A0, B0, x, y)

xx = np.linspace(dx / 2, Lx - dx / 2, Nx)
yy = np.linspace(dx / 2, Ly - dx / 2, Ny)
x, y = np.meshgrid(xx, yy)

φ_ini = get_phi(initial_conc(x, y))


def run(timesteps, dt_conc):
    τ_fin = np.ceil(get_phi_time(timesteps))
    dt_phi = get_phi_time(dt_conc)

    φ = np.reshape(φ_ini, (1, Nx, Ny))

    print("t={:,d} -> τ={:,d}".format(int(timesteps), int(τ_fin)))

    # Write to disk uniformly in logarithmic space
    checkpoints = np.logspace(
        0.0,
        np.log(τ_fin),
        num=100,
        base=np.e
    )

    τ = 0.0
    time_energy = []
    write_plot(φ[0])

    progress = tqdm(CheckpointStepper(start=0,
                                      stops=checkpoints,
                                      stop=τ_fin),
                    unit="step",
                    total=len(checkpoints),
                    bar_format='{l_bar}{bar:40}{r_bar}{bar:-40b}')

    for check in progress:
        for step in FixedStepper(start=check.begin,
                                 stop=check.end,
                                 size=dt_phi):
            label = "[{:>13.2f} .. {:>13.2f})".format(τ, check.end)
            progress.set_description(label)

            φ = solve(φ, delta_t=step.size).compute()
            τ += step.size

            time_energy.append([time.time() - startTime,
                                get_conc_time(τ),
                                calc_f_total(φ, dx, κ),
                                proc.memory_info().rss / 1024])

            _ = step.succeeded()

        gc.collect()

        write_plot(φ[0], time_energy[-1][1])

        _ = check.succeeded()

    return np.array(time_energy)


time_energy = run(t_fin, dt)

np.savetxt(f"pymks/{variant}/free_energy.csv",
           time_energy,
           delimiter=",",
           header="wall_time,time,free_energy,mem_KB",
           fmt="%.12f")

plt.figure(figsize=(10,8))
plt.xlabel(r"simulation time $t$")
plt.ylabel(r"free energy $\mathcal{F}$")
plt.loglog(time_energy[:, 1], time_energy[:, 2])
plt.savefig(f"pymks/{variant}/energy.png",
            bbox_inches="tight", dpi=400)
plt.close()

plt.figure(figsize=(10,8))
plt.xlabel(r"simulation time $t$")
plt.ylabel(r"memory / [KB]")
plt.plot(time_energy[:, 0], time_energy[:, 3])
plt.savefig(f"pymks/{variant}/memory.png",
            bbox_inches="tight", dpi=400)
plt.close()

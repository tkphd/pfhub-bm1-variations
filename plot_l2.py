#!/usr/bin/env python3

"""
Perform spectral interpolation of grid data
and compute L2 norms by remapping the domain
from [0, L] to [0, 2π] and using the periodic
sinc function,

$$
S_n = \frac{h}{L} \frac{\sin(\pi x / h)}{\tan(x / 2)}
$$
"""

from argparse import ArgumentParser
import gc
import glob
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import os
from parse import parse
import sys
import time

# import from `spectral.py` in same folder as the script
sys.path.append(os.path.dirname(__file__))

# from spectral import SpectralInterpolant as Interpolant
from spectral import CoincidentInterpolant as Interpolant

# reset color cycle for 16 lines
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.hsv(np.linspace(0, 1, 16)))

# parse command-line flags
parser = ArgumentParser()
parser.add_argument("--dx",   type=float, help="Candidate Gold Standard resolution")
parser.add_argument("--dt",   type=float, help="Timestep of interest")
parser.add_argument("--time", type=int,   help="Time slice of interest",
                              nargs="+")
args = parser.parse_args()

variant = os.path.basename(os.getcwd())

def elapsed(stopwatch):
    return np.round(time.time() - stopwatch, 2)


def sim_details(dir):
    _, dx = parse("dt{:6.4f}_dx{:6.4f}", dir)
    Nx = np.rint(200. / dx).astype(int)
    slices = sorted(glob.glob(f"{dir}/c_*.npz"))
    _, t_max = parse("{}/c_{}.npz", slices[-1])

    return float(dx), int(Nx), int(t_max)


def log_hn(h, n):
    """
    Support function for plotting 𝒪(hⁿ)
    """
    return n * np.log(h)


# get "gold standard" info
goldir = f"dt{args.dt:6.04f}_dx{args.dx:6.04f}"
gold_h, gold_N, gold_T = sim_details(goldir)

if gold_N % 2 != 0:
    raise ValueError("Reference mesh size is not even!")

print(f"Candidate with h={gold_h} has reached t={gold_T}\n")


# set output image file
png = f"norm_{variant}_dt{args.dt:6.04f}.png"

plt.figure(1, figsize=(10,8))
plt.title(f"IC: {variant}")
plt.xlabel("Mesh size $N_x$ / [a.u.]")
plt.ylabel("L2 norm, $||\\Delta c||_2$ / [a.u.]")

# plt.ylim([1, 50])

# Interpolate!

sinterp = Interpolant(gold_N, gold_N)

jobs = {}

for job in sorted(glob.glob(f"dt{args.dt:6.04f}_dx?.????")):
    stats = sim_details(job)
    if stats[0] > gold_h:
        jobs[job] = stats

times = np.unique(np.concatenate([np.array([0], dtype=int),
                                  np.array(args.time, dtype=int)]))
times = times[times <= gold_T]

for t in times:
    resolution = []
    norm = []

    with np.load(f"{goldir}/c_{t:08d}.npz") as npz:
        gold_c = npz["c"]

    for jobdir, (job_h, job_N, job_T) in jobs.items():
        print(f"Interpolating {jobdir} @ t={t:,d}")

        refined = f"{jobdir}/k_{t:08d}_h{gold_h:6.04f}.npz"
        job_refined = None

        if not os.path.exists(refined):
            try:
                job_refined = np.zeros((gold_N, gold_N), dtype=float)

                with np.load(f"{jobdir}/c_{t:08d}.npz") as npz:
                    job_c = npz["c"]

                watch = time.time()
                job_refined = sinterp.upsample(job_c)

                np.savez_compressed(refined, c=job_refined)

            except FileNotFoundError:
                job_refined = None
        else:
            with np.load(refined) as npz:
                job_refined = npz["c"]

        if job_refined is not None:
            print("    L2: ", end="")
            resolution.append(job_N)
            norm.append(gold_h * LA.norm(gold_c - job_refined))
            print(f"{norm[-1]:9.03e}")

            refined_png = refined.replace("npz", "png")
            if not os.path.exists(refined_png):
                plt.figure(2, figsize=(10, 8))
                plt.title(f"$\\Delta x={job_h},\\ \\Delta t={args.dt}\\ @\\ t={t:,d}$")
                plt.xlabel("$k_x$ / [a.u.]")
                plt.ylabel("$k_y$ / [a.u.]")
                fft_refined = np.fft.fftshift(np.fft.fft2(job_refined))
                plt.colorbar(plt.imshow(fft_refined.real, norm="asinh", cmap="gray",
                                        interpolation=None, origin="lower"))
                plt.savefig(refined_png, dpi=400, bbox_inches="tight")
                plt.close()

        gc.collect()

    plt.figure(1)
    plt.loglog(resolution, norm, marker="o", label=f"$t={t:,d}$")

    print()

ylim = plt.ylim()
plt.ylim(ylim)

h = np.linspace(2 * gold_h, 6.25, 100)
plt.loglog(200/h, h**2, color="silver", label=r"$\mathcal{O}(h^2)$")
plt.loglog(200/h, h**4, color="silver", linestyle="dashed", label=r"$\mathcal{O}(h^4)$")

plt.legend(loc="best")
plt.savefig(png, dpi=400, bbox_inches="tight")

print(f"Saved image to {png}")

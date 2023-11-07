#!/usr/bin/env python3

"""
Perform spectral interpolation of grid data
and compute L2 norms by remapping the domain
from [0, L] to [0, 2π] and zero-padding in
reciprocal space. For details, see
`fourier-interpolation.ipynb` in this repo.
"""

from argparse import ArgumentParser
import gc
import glob
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import os
from parse import parse
import time
try:
    from rich import print
except ImportError:
    pass

# import from `spectral.py` in same folder as the script
import sys
sys.path.append(os.path.dirname(__file__))

from spectral import FourierInterpolant as Interpolant
from spectral import MidpointNormalize, autocorrelation, radial_profile


def elapsed(stopwatch):
    """
    Return the number of whole seconds elapsed since the mark
    """
    return np.ceil(time.time() - stopwatch).astype(int)


def sim_details(iodir):
    _, dx = parse("{}{:08.04f}", iodir)
    Nx = np.rint(200. / dx).astype(int)
    slices = sorted(glob.glob(f"{iodir}/c_*.npz"))
    _, t_max = parse("{}/c_{:08d}.npz", slices[-1])

    return float(dx), int(Nx), int(t_max)


def log_hn(h, n, b=np.log(1000)):
    """
    Support function for plotting 𝒪(hⁿ) on a log-log scale:
      log(y) = n log(h) + b
             = log(hⁿ) + b
          y  = hⁿ exp(b)

    Inputs
    ------
    h: array of dx values
    n: order of accuracy
    b: intercept
    """
    return np.exp(b) * h**n


variant = os.path.basename(os.getcwd())

# reset color cycle for 50 lines
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.hsv(np.linspace(0, 1, 50)))

# parse command-line flags
parser = ArgumentParser()
parser.add_argument("--dx", type=float,
                            help="Candidate Gold Standard resolution")
parser.add_argument("--dt", type=float,
                            help="Timestep of interest")
args = parser.parse_args()

# get "gold standard" info
goldir = glob.glob(f"*dx{args.dx:08.04f}")[0]
gold_h, gold_N, gold_T = sim_details(goldir)

if gold_N % 2 != 0:
    raise ValueError("Reference mesh size is not even!")

print(f"=== {variant}/{goldir} has reached t={gold_T} ===\n")

# set output image file
png = f"norm_{variant}_dt{args.dt:6.04f}.png"

plt.figure(1, figsize=(10, 8))
plt.title(f"\"{variant.capitalize()}\" IC")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Mesh size $N_x$ / [a.u.]")
plt.ylabel("$\\ell^2$ norm, $||\\Delta c||_2$ / [a.u.]")
plt.ylim([5e-14, 5e3])

# plot lines for known orders of accuracy
h = np.linspace(2 * gold_h, 50, 100)
N = 200 / h

plt.plot(N, log_hn(N, -1, np.log(4e3)), color="silver",
         label=r"$\mathcal{O}(h^1)$", zorder=0, linestyle="dotted")
plt.plot(N, log_hn(N, -2, np.log(6e3)), color="silver",
         label=r"$\mathcal{O}(h^2)$", zorder=0, linestyle="solid")
plt.plot(N, log_hn(N, -3, np.log(8e3)), color="silver",
         label=r"$\mathcal{O}(h^3)$", zorder=0, linestyle="dashdot")

# Interpolate!

sinterp = Interpolant((gold_N, gold_N))

jobs = {}

for job in sorted(glob.glob("*dx???.????")):
    stats = sim_details(job)
    if stats[0] > gold_h:
        jobs[job] = stats


for golden in sorted(glob.glob(f"{goldir}/c_????????.npz")):
    resolutions = []
    norms = []

    _, t = parse("{}/c_{:d}.npz", golden)

    print(f"  Interpolating {variant.capitalize()}s @ t = {t:,d} / {gold_T:,d}")

    with np.load(golden) as npz:
        gold_c = npz["c"]

    gold_stats = f"{goldir}/stats_{t:08d}.npz"

    if not os.path.exists(gold_stats):
        # compute autocorrelation, radial-avg
        gold_cor = autocorrelation(gold_c)
        gold_r, gold_μ = radial_profile(gold_cor)
        gold_r = gold_h * np.array(gold_r)
        np.savez_compressed(gold_stats, corr=gold_cor, r=gold_r, μ=gold_μ)

    for jobdir, (job_h, job_N, job_T) in jobs.items():
        terpdir = f"{jobdir}/interp"
        if not os.path.exists(terpdir):
            os.mkdir(terpdir)

        refined = f"{terpdir}/k_{t:08d}_h{gold_h:6.04f}.npz"
        refined_png = refined.replace("npz", "png")

        job_stats = f"{terpdir}/stats_{t:08d}_h{gold_h:6.04f}.npz"

        job_refined = None
        ell_two = None
        watch = None

        if not os.path.exists(refined):
            print(f"    {jobdir}:", end=" ")
            try:
                with np.load(f"{jobdir}/c_{t:08d}.npz") as npz:
                    job_c = npz["c"]

                startNorm = time.time()
                job_refined = sinterp.upsample(job_c)
                ell_two = LA.norm(gold_c - job_refined)
                np.savez_compressed(refined, c=job_refined, l2=ell_two)
                watch = elapsed(startNorm)
                print(f"ℓ² = {ell_two:.02e}  ({watch:2d} s)")
            except FileNotFoundError:
                job_refined = None
                ell_two = None
                print("failed.")

        with np.load(refined) as npz:
            ell_two = npz["l2"]

            if ell_two is not None:
                resolutions.append(job_N)
                norms.append(ell_two)

            if not os.path.exists(refined_png):
                if job_refined is None:
                    job_refined = npz["c"]
                fig, axs = plt.subplots(1, 2, figsize=(10, 4),
                                        constrained_layout=True, sharex=True, sharey=True)

                fig.suptitle(
                    f"\"{variant.capitalize()}\" IC: $\\Delta x={job_h}\\ @\\ t={t:,d}$")
                axs[0].set_xlabel("$x$ / [a.u.]")
                axs[0].set_ylabel("$y$ / [a.u.]")

                c_min = np.amin(job_refined)
                c_avg = np.average(job_refined)
                c_max = np.amax(job_refined)
                c_nrm = MidpointNormalize(midpoint=c_avg, vmin=c_min, vmax=c_max)

                axs[0].set_title(r"$c$")
                fig.colorbar(
                    axs[0].imshow(job_refined, cmap="coolwarm", clim=(c_min, c_max),
                                  norm=c_nrm, interpolation=None, origin="lower")
                )

                diff_c = np.absolute(job_refined - gold_c)
                axs[1].set_title(r"$(\Delta c)^2$")

                fig.colorbar(
                    axs[1].imshow(diff_c, norm="log", cmap="twilight_shifted",
                                  interpolation=None, origin="lower")
                )
                fig.savefig(refined_png, dpi=400, bbox_inches="tight")
                plt.close(fig)

            if not os.path.exists(job_stats):
                # compute autocorrelation, radial-avg
                if job_refined is None:
                    job_refined = npz["c"]
                job_cor = autocorrelation(job_refined)
                job_r, job_μ = radial_profile(job_cor)
                job_r = job_h * np.array(job_r)
                np.savez_compressed(job_stats, corr=job_cor, r=job_r, μ=job_μ)

        gc.collect()

    plt.figure(1)
    plt.plot(resolutions, norms, marker="o", label=f"$t={t:,d}$")


plt.legend(ncol=6, loc="best")
plt.savefig(png, dpi=400, bbox_inches="tight")
print(f"\n  Saved image to {png}\n")

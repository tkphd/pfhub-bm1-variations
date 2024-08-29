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
import pyfftw

import os
from parse import compile
import time
try:
    from rich import print
except ImportError:
    pass
import sys
from zipfile import BadZipFile

# import from `spectral.py` in same folder as the script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from spectral.viz import FourierInterpolant as Interpolant
from spectral.viz import MidpointNormalize, autocorrelation, radial_profile

# my goofy folder naming conventions
job_pattern = "dx???.????"

parse_dx  = compile("{prefix}x{dx:8f}")
parse_npz = compile("{prefix}/c_{t:d}.npz")

# threaded FFTW shenanigans
pyfftw.config.NUM_THREADS = float(os.environ["OMP_NUM_THREADS"])

def elapsed(stopwatch):
    """
    Return the number of whole seconds elapsed since the mark
    """
    return np.ceil(time.time() - stopwatch).astype(int)


def sim_details(iodir):
    dx = parse_dx.parse(iodir)["dx"]
    Nx = np.rint(200. / dx).astype(int)

    slices = sorted(glob.glob(f"{iodir}/c_*.npz"))

    t_max = parse_npz.parse(slices[-1])["t"]

    return {
        "dx": float(dx),
        "Nx": int(Nx),
        "t_max": int(t_max)
    }


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
args = parser.parse_args()

dirs = sorted(glob.glob(job_pattern))

# get "gold standard" info
goldir = dirs[0]
gold_par = sim_details(goldir)

gold_h = gold_par["dx"]
gold_N = gold_par["Nx"]
gold_T = gold_par["t_max"]

if gold_N % 2 != 0:
    raise ValueError("Reference mesh size is not even!")

print(f"=== {variant}/{goldir} has reached t={gold_T} ===\n")

# set output image file
png = f"norm_{variant}_PIDStepper.png"

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

for job in dirs:
    stats = sim_details(job)
    if stats["dx"] > gold_h:
        jobs[job] = stats

gold_npzs = sorted(glob.glob(f"{goldir}/c_????????.npz"))

for golden in gold_npzs:
    resolutions = []
    norms = []

    try:
        t = parse_npz.parse(golden)["t"]

        print(f"  Interpolating {variant.capitalize()}s @ t = {t:,d} / {gold_T:,d}")

        with np.load(golden) as npz:
            gold_c = npz["c"]

            gold_stats = f"{goldir}/stats_{t:08d}.npz"
    except BadZipFile:
        print(f"Unable to load {golden}")
        sys.exit()

    if not os.path.exists(gold_stats):
        # compute autocorrelation, radial-avg
        gold_cor = autocorrelation(gold_c)
        gold_r, gold_μ = radial_profile(gold_cor)
        gold_r = gold_h * np.array(gold_r)
        np.savez_compressed(gold_stats, corr=gold_cor, r=gold_r, μ=gold_μ)

    for jobdir, job_par in jobs.items():
        job_h = job_par["dx"]
        job_N = job_par["Nx"]
        job_T = job_par["t_max"]
        terpdir = f"{jobdir}/interp"
        if not os.path.exists(terpdir):
            os.mkdir(terpdir)

        refined = f"{terpdir}/k_{t:08d}_h{gold_h:6.04f}.npz"
        refined_png = refined.replace("npz", "png")

        job_stats = f"{terpdir}/stats_{t:08d}_h{gold_h:6.04f}.npz"

        job_refined = None
        ell_two = None
        watch = None
        startNorm = time.time()

        if not os.path.exists(refined):
            print(f"    {jobdir}:", end=" ")
            try:
                with np.load(f"{jobdir}/c_{t:08d}.npz") as npz:
                    job_c = npz["c"]

                job_refined = sinterp.upsample(job_c)
                ell_two = LA.norm(gold_c - job_refined)
                np.savez_compressed(refined, c=job_refined, l2=ell_two)
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

        watch = elapsed(startNorm)
        print(f"      ℓ² = {ell_two:.02e}  ({watch:2d} s)")

        gc.collect()

    plt.figure(1)
    plt.plot(resolutions, norms, marker="o", label=f"$t={t:,d}$")

if len(gold_npzs) < 50:
    plt.legend(ncol=3, loc=3, fontsize=9)
plt.savefig(png, dpi=400, bbox_inches="tight")
print(f"\n  Saved image to {png}\n")

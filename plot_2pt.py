#!/usr/bin/env python3

"""
Perform spectral interpolation of grid data by remapping the domain
from [0, L] to [0, 2π] and zero-padding in reciprocal space.
(Do not FFT-shift.)
Use the FFT to generate the power spectrum, $\tilde{y}^*\tilde{y}$.
Transform the power spectrum back to real space, producing the
autocorrelation (aka 2-pt stats).
Compute the l² norm of the autocorrelation against the gold-standard
autocorrelation. Publish, win fame and admiration, achieve knighthood.
"""

from argparse import ArgumentParser
import gc
import glob
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

import os
from parse import compile
import time
try:
    from rich import print
except ImportError:
    pass
import sys
import zipfile

# import from `spectral.py` in same folder as the script
sys.path.append(os.path.dirname(__file__))

from spectral import FourierInterpolant as Interpolant
from spectral import MidpointNormalize, radial_profile

# my goofy folder naming conventions
old_pattern = "dt?.????_dx???.????"
new_pattern = "dx???.????"

parse_dt  = compile("dt{dt:6f}{suffix}")
parse_dx  = compile("{prefix}x{dx:8f}")
parse_dtx = compile("dt{dt:6f}_dx{dx:8f}")
parse_npz = compile("{prefix}/c_{t:8d}.npz")

def correlate(data):
    """Compute the auto-correlation / 2-point statistics of a field variable"""
    signal = data - data.mean()
    fft = np.fft.fftn(signal)
    psd = fft * np.conjugate(fft)
    return np.fft.ifftn(psd).real / (np.var(signal) * signal.size)


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
parser.add_argument("--dt", type=float,
                            help="Timestep of interest")
args = parser.parse_args()

dirs = sorted(glob.glob(old_pattern) + glob.glob(new_pattern))

# get "gold standard" info
goldir = dirs[0]
gold_par = sim_details(goldir)

gold_h = gold_par["dx"]
gold_N = gold_par["Nx"]
gold_T = gold_par["t_max"]

gold_freq = 2 * np.pi * np.fft.fftfreq(gold_N, d=gold_h)
nyquist = gold_freq.max() / 2

if gold_N % 2 != 0:
    raise ValueError("Reference mesh size is not even!")

print(f"=== {variant}/{goldir} has reached t={gold_T:,d} ===\n")

# set output image file
png = f"auto_{variant}_dt{args.dt:6.04f}.png"

plt.figure(1, figsize=(10, 8))
plt.title(f"\"{variant.capitalize()}\" IC: Auto-Correlation")
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

# gold_npzs = sorted(glob.glob(f"{goldir}/c_????????.npz"))

gold_npzs = [f"{goldir}/c_{x:08d}.npz" for x in range(11)]

for golden in gold_npzs:
    t = parse_npz.parse(golden)["t"]

    resolutions = []
    norms = []

    gold_cor = None
    gold_c = None

    try:
        with np.load(golden) as npz:
            gold_c = npz["c"]
    except FileNotFoundError:
        gold_c = None
        pass
    except zipfile.BadZipFile:
        gold_c = None
        pass

    if gold_c is not None:
        print(f"  Interpolating {variant.capitalize()}s @ t = {t:,d} / {gold_T:,d}")

        gold_stats = f"{goldir}/interp/stats_{t:08d}.npz"
        gold_png = gold_stats.replace("npz", "png")

        if not os.path.exists(f"{goldir}/interp"):
            os.mkdir(f"{goldir}/interp")

        if not os.path.exists(gold_stats):
            # compute autocorrelation, radial-avg
            gold_cor = correlate(gold_c)
            gold_r, gold_μ = radial_profile(gold_cor)
            gold_r = gold_h * np.array(gold_r)
            np.savez_compressed(gold_stats,
                                t=t,
                                dx=gold_h,
                                auto=gold_cor,
                                r=gold_r,
                                μ=gold_μ)
        else:
            try:
                with np.load(gold_stats) as npz:
                    gold_cor = npz["corr"]
            except FileNotFoundError or zipfile.BadZipFile:
                gold_cor = None
                pass

        if gold_cor is not None and not os.path.exists(gold_png):
            fig, ax = plt.subplots(1, 1, figsize=(5, 4),
                                   constrained_layout=True, sharex=True, sharey=True)

            fig.suptitle(
                f"\"{variant.capitalize()}\" IC: Autocorr, $\\Delta x={gold_h}\\ @\\ t={t:,d}$")
            ax.set_xlabel("$x$ / [a.u.]")
            ax.set_ylabel("$y$ / [a.u.]")

            c_min = np.amin(gold_cor)
            c_avg = np.average(gold_cor)
            c_max = np.amax(gold_cor)
            c_nrm = MidpointNormalize(midpoint=c_avg, vmin=c_min, vmax=c_max)

            ax.set_title(r"$c$")
            fig.colorbar(
                ax.imshow(gold_cor, cmap="coolwarm", clim=(c_min, c_max),
                          norm=c_nrm, interpolation=None, origin="lower")
            )

            fig.savefig(gold_png, dpi=400, bbox_inches="tight")
            plt.close(fig)


        for jobdir, job_par in jobs.items():
            startNorm = time.time()
            watch   = None

            job_h = job_par["dx"]
            job_N = job_par["Nx"]
            job_T = job_par["t_max"]

            job_refined = None
            job_cor = None
            ell_two = None

            terpdir = f"{jobdir}/interp"
            if not os.path.exists(terpdir):
                os.mkdir(terpdir)

            refined   = f"{terpdir}/k_{t:08d}_h{gold_h:6.04f}.npz"
            job_stats = f"{terpdir}/stats_{t:08d}_h{gold_h:6.04f}.npz"
            stats_png = job_stats.replace("npz", "png")

            print(f"    {jobdir}:", end=" ")

            if not os.path.exists(refined):
                try:
                    with np.load(f"{jobdir}/c_{t:08d}.npz") as npz:
                        job_c = npz["c"]

                    job_refined = sinterp.upsample(job_c)
                    ell_two = LA.norm(gold_c - job_refined)
                    np.savez_compressed(refined,
                                        t=t,
                                        c=job_refined,
                                        l2=ell_two)
                except FileNotFoundError or zipfile.BadZipFile:
                    job_refined = None
                    print("failed.")
                    pass
            else:
                try:
                    with np.load(refined) as npz:
                        job_refined = npz["c"]
                except FileNotFoundError or zipfile.BadZipFile:
                    job_refined = None
                    print("failed.")
                    pass

            if (job_refined is not None) and (not os.path.exists(job_stats)):
                # compute autocorrelation, radial-avg
                job_cor = correlate(job_refined)
                job_r, job_μ = radial_profile(job_cor)
                job_r = job_h * np.array(job_r)
                np.savez_compressed(job_stats,
                                    t=t,
                                    dx=job_h,
                                    auto=job_cor,
                                    r=job_r,
                                    μ=job_μ)
            else:
                try:
                    with np.load(job_stats) as npz:
                        job_cor = npz["corr"]
                except FileNotFoundError or zipfile.BadZipFile:
                    job_cor = None
                    print("failed.")
                    pass

            if gold_cor is not None and job_cor is not None:
                ell_two = LA.norm(gold_cor - job_cor)
                resolutions.append(job_N)
                norms.append(ell_two)

            if job_cor is not None and not os.path.exists(stats_png):
                fig, ax = plt.subplots(1, 1, figsize=(5, 4),
                                       constrained_layout=True, sharex=True, sharey=True)

                fig.suptitle(
                    f"\"{variant.capitalize()}\" IC: Autocorr, $\\Delta x={job_h}\\ @\\ t={t:,d}$")
                ax.set_xlabel("$x$ / [a.u.]")
                ax.set_ylabel("$y$ / [a.u.]")

                c_min = np.amin(job_cor)
                c_avg = np.average(job_cor)
                c_max = np.amax(job_cor)
                c_nrm = MidpointNormalize(midpoint=c_avg,
                                          vmin=c_min,
                                          vmax=c_max)

                ax.set_title(r"$c$")
                fig.colorbar(
                    ax.imshow(job_cor, cmap="coolwarm", clim=(c_min, c_max),
                              norm=c_nrm, interpolation=None, origin="lower")
                )

                fig.savefig(stats_png, dpi=400, bbox_inches="tight")
                plt.close(fig)

            watch = elapsed(startNorm)
            print(f" ℓ² = {ell_two:.02e}  ({watch:2d} s)")

            gc.collect()

    plt.figure(1)
    if len(resolutions) > 0:
        plt.plot(resolutions, norms, marker="o", label=f"$t={t:,d}$")

if len(gold_npzs) < 50:
    plt.legend(ncol=3, loc=3, fontsize=9)
plt.savefig(png, dpi=400, bbox_inches="tight")
plt.savefig(f"../auto_{variant}.png", dpi=400, bbox_inches="tight")

print(f"\n  Saved image to {png}\n")

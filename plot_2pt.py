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
import zlib

# import from `spectral.py` in same folder as the script
sys.path.append(os.path.dirname(__file__))

from spectral import FourierInterpolant, log_hn, radial_profile

parse_dt  = compile("dt{dt:8f}{suffix}")
parse_dx  = compile("{prefix}x{dx:8f}")
parse_dtx = compile("dt{dt:8f}_dx{dx:8f}")
parse_npz = compile("{prefix}/c_{t:8d}.npz")


def correlate(data):
    """Compute the auto-correlation / 2-point statistics of a field variable"""
    signal = data - data.mean()
    fft = np.fft.rfftn(signal)
    psd = fft * np.conjugate(fft)
    return np.fft.irfftn(psd).real / (np.var(signal) * signal.size)


def elapsed(stopwatch):
    """
    Return the number of whole seconds elapsed since the mark
    """
    return np.ceil(time.time() - stopwatch).astype(int)


def sim_details(iodir):
    dx = parse_dx.parse(iodir)["dx"]
    Nx = np.rint(200. / dx).astype(int)

    slices = sorted(glob.glob(f"{iodir}/c_*.npz"))

    t_max = parse_npz.parse(slices[-1])["t"] if len(slices) > 1 else 0

    return {
        "dx": float(dx),
        "Nx": int(Nx),
        "t_max": int(t_max)
    }


def upsampled(c_npz, k_npz, job_h, mesh_h=0.0625, interpolant=None):
    hi_res = None
    hi_cor = None
    hi_fft = None
    cor_r  = None
    cor_μ  = None
    mesh_N = 200 // mesh_h  # int(3200 * mesh_h / 0.0625)

    if interpolant is None:
        interpolant = FourierInterpolant((mesh_N, mesh_N))

    if not os.path.exists(k_npz):
        # Upsample and save spectrally interpolated mesh data
        try:
            with np.load(c_npz) as npz:
                lo_res = npz["c"]

            hi_res = interpolant.upsample(lo_res)
            signal = hi_res - hi_res.mean()
            hi_fft = np.fft.fftn(signal)
            hi_psd = hi_fft * np.conjugate(hi_fft)
            hi_cor = np.fft.ifftn(hi_psd).real / (np.var(signal) * signal.size)
            cor_r, cor_μ = radial_profile(hi_cor)
            cor_r = gold_h * np.array(cor_r)

            np.savez_compressed(k_npz,
                                t=t,
                                dx=mesh_h,
                                c=hi_res,
                                k=hi_fft,
                                p=hi_psd,
                                a=hi_cor,
                                r=cor_r,
                                μ=cor_μ)
        except FileNotFoundError or zipfile.BadZipFile or zlib.error:
            print("failed (no data).")
            pass
    else:
        try:
            with np.load(k_npz) as npz:
                hi_res = npz["c"]
                hi_cor = npz["a"]
        except FileNotFoundError or zipfile.BadZipFile or zlib.error:
            print("failed (bad stored data).")
            pass

    hi_png = k_npz.replace("npz", "png")

    if not os.path.exists(hi_png):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4),
                                constrained_layout=True, sharex=False, sharey=False)
        fig.suptitle(
            f"\"{variant.capitalize()}\" IC: $\\Delta x={job_h}\\ @\\ t={t:,d}$")

        axs[0].set_xlabel("$n_x$ / [a.u.]")
        axs[0].set_ylabel("$n_y$ / [a.u.]")
        axs[0].set_title("composition")

        axs[1].set_xlabel("$x$ / [a.u.]")
        axs[1].set_ylabel("$\\sigma$ / [a.u.]")
        axs[1].set_title("power spectrum")

        if cor_r is None or cor_μ is None:
            try:
                with np.load(k_npz) as npz:
                    cor_r = npz["r"]
                    cor_μ = npz["μ"]
            except FileNotFoundError:
                pass


        if hi_res is not None:
            fig.colorbar(
                axs[0].imshow(hi_res, cmap="coolwarm", clim=(0.3, 0.7),
                              interpolation=None, origin="lower")
            )

        if cor_r is not None:
            axs[1].plot(cor_r, cor_μ)

        fig.savefig(hi_png, dpi=400, bbox_inches="tight")
        plt.close(fig)

    return hi_res, hi_cor


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

dirs = sorted(glob.glob("dt?.??????_dx???.????"))

# get "gold standard" info
goldir = dirs[0]
gold_par = sim_details(goldir)

gold_h = gold_par["dx"]
gold_N = gold_par["Nx"]
gold_T = gold_par["t_max"]

if gold_N % 2 != 0:
    raise ValueError("Reference mesh size is not even!")

mesh_h = 2**-4  # min(2**(-4), gold_h)  # 0.00625 == 2**-4
mesh_N = gold_N  # int(gold_N * gold_h / mesh_h)
sinterp = FourierInterpolant((mesh_N, mesh_N))

print(f"=== {variant}/{goldir} has reached t={gold_T:,d} ===\n")

# set output image file
png = f"auto_{variant}_dt{args.dt:8.06f}.png"

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
         label=r"$\mathcal{O}(h^3)$", zorder=0, linestyle="dashed")
plt.plot(N, log_hn(N, -4, np.log(1e4)), color="silver",
         label=r"$\mathcal{O}(h^4)$", zorder=0, linestyle="dashdot")


# === Interpolate! ===

if not os.path.exists(f"{goldir}/interp"):
    os.mkdir(f"{goldir}/interp")

jobs = {}

for job in dirs:
    stats = sim_details(job)
    if stats["dx"] > gold_h:
        jobs[job] = stats

gold_npzs = sorted(glob.glob(f"{goldir}/c_????????.npz"))

# gold_npzs = [f"{goldir}/c_{x:08d}.npz" for x in range(11)]

for golden in gold_npzs:
    t = parse_npz.parse(golden)["t"]
    kolden = f"{goldir}/interp/k_{t:08d}.npz"

    resolutions = []
    norms = []

    _, gold_cor = upsampled(golden, kolden, gold_h, mesh_h, sinterp)

    if gold_cor is not None:
        print(f"  Interpolating {variant.capitalize()}s @ t = {t:,d} / {gold_T:,d}")

        for jobdir, job_par in jobs.items():
            startNorm = time.time()
            print(f"    {jobdir}:", end=" ")

            terpdir = f"{jobdir}/interp"
            if not os.path.exists(terpdir):
                os.mkdir(terpdir)

            job_N = job_par["Nx"]
            job_h = job_par["dx"]

            job_c_npz = f"{jobdir}/c_{t:08d}.npz"
            job_k_npz = f"{jobdir}/interp/c_{t:08d}.npz"

            _, job_cor = upsampled(job_c_npz, job_k_npz, job_h, mesh_h, sinterp)

            if gold_cor is not None and job_cor is not None:
                ell_two = LA.norm(gold_cor - job_cor)

                resolutions.append(job_N)
                norms.append(ell_two)

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

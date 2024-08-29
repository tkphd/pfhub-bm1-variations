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
import pandas as pd
import pyfftw.builders as FFTW

import os
import sys
from parse import compile
import time
try:
    from rich import print
except ImportError:
    pass
import zipfile
import zlib

# import from `spectral.py` in same folder as the script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from spectral.viz import FourierInterpolant, L, log_hn, radial_profile

parse_dx  = compile("{prefix}x{dx:8f}")
parse_npz = compile("{prefix}/c_{t:8d}.npz")

def elapsed(stopwatch):
    """
    Return the number of whole seconds elapsed since the mark
    """
    return np.ceil(time.time() - stopwatch).astype(int)


def sim_details(iodir):
    dx = parse_dx.parse(iodir)["dx"]
    Nx = int(L / dx)

    slices = sorted(glob.glob(f"{iodir}/c_*.npz"))

    t_max = parse_npz.parse(slices[-1])["t"] if len(slices) > 1 else 0

    return {
        "dx": float(dx),
        "Nx": int(Nx),
        "t_max": int(t_max)
    }


def upsampled(c_npz, k_npz, job_h, mesh_h=0.0625, interpolant=None):
    mesh_N = int(L // mesh_h)

    hi_res = None
    hi_cor = None
    hi_fft = None
    cor_r  = None
    cor_μ  = None

    if interpolant is None:
        interpolant = FourierInterpolant((mesh_N, mesh_N))

    if not os.path.exists(k_npz):
        # Upsample and save spectrally interpolated mesh data
        try:
            with np.load(c_npz) as npz:
                lo_res = npz["c"]

            hi_res = interpolant.upsample(lo_res)
            signal = hi_res - hi_res.mean()
            sig_fft = FFTW.fftn(signal.copy())

            hi_fft = sig_fft()
            hi_psd = hi_fft * np.conjugate(hi_fft)

            cor_ift = FFTW.ifftn(hi_psd.copy())
            hi_cor = cor_ift().real / (np.var(signal) * signal.size)
            cor_r, cor_μ = radial_profile(hi_cor)
            cor_r = gold_h * np.array(cor_r)

            np.savez_compressed(k_npz,
                                t=t,
                                dx=mesh_h,
                                c=hi_res,
                                r=cor_r,
                                μ=cor_μ)
        except FileNotFoundError or zipfile.BadZipFile or zlib.error:
            print("failed (no data).")
            pass
    else:
        try:
            with np.load(k_npz) as npz:
                hi_res = npz["c"]
        except FileNotFoundError or zipfile.BadZipFile or zlib.error or KeyError:
            print(f"failed, bad data in {k_npz}")
            pass

    return hi_res


def dataviz(c_npz, job_h, t, png):
    job_c = None
    job_r  = None
    job_μ  = None

    if not os.path.exists(png):
        with np.load(c_npz) as npz:
            job_c = npz["c"]
            job_r = npz["r"]
            job_μ = npz["μ"]

        fig, axs = plt.subplots(1, 2, figsize=(7.5, 3),
                                constrained_layout=True,
                                sharex=False, sharey=False)
        fig.suptitle(
            f"\"{variant.capitalize()}\" IC: $\\Delta x={job_h}\\ @\\ t={t:,d}$")

        axs[0].set_xlabel("$n_x$ / [a.u.]")
        axs[0].set_ylabel("$n_y$ / [a.u.]")
        axs[0].set_title("composition")

        axs[1].set_xlabel("$x$ / [a.u.]")
        axs[1].set_ylabel("$\\sigma$ / [a.u.]")
        axs[1].set_title("power spectrum")

        if job_c is not None:
            job_L = job_c.shape[0] * job_h
            fig.colorbar(
                axs[0].imshow(job_c,
                              cmap="coolwarm",
                              clim=(0.3, 0.7),
                              extent=(0, job_L, 0, job_L),
                              interpolation=None,
                              origin="lower")
            )

        if job_r is not None:
            axs[1].plot(job_r, job_μ)
            axs[1].set_xlim((0, L//2))

        fig.savefig(png, dpi=400, bbox_inches="tight")
        plt.close(fig)

variant = os.path.basename(os.getcwd())

# reset color cycle for 50 lines
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.hsv(np.linspace(0, 1, 50)))

# parse command-line flags
parser = ArgumentParser()
parser.add_argument("--dx",
                    type=float,
                    help="Candidate Gold Standard resolution")
args = parser.parse_args()

dirs = sorted(glob.glob("dx???.????"))

# get "gold standard" info
goldir = dirs[0]
gold_par = sim_details(goldir)

gold_h = gold_par["dx"]
gold_N = gold_par["Nx"]
gold_T = gold_par["t_max"]
gold_L = gold_h * gold_N

if gold_N % 2 != 0:
    raise ValueError("Reference mesh size is not even!")

if not np.isclose(L, gold_L):
    raise ValueError("Mismatched domain sizes! How‽")

mesh_h = 2**-4  # 0.0625 == 2**-4
mesh_N = int(gold_N)
sinterp = FourierInterpolant((mesh_N, mesh_N))

print(f"=== {variant}/{goldir} has reached t={gold_T:,d} ===\n")

# set output image file
png = f"../img/auto_{variant}_adaptive.png"

plt.figure(1, figsize=(10, 8))
plt.title(f"\"{variant.capitalize()}\" IC: Auto-Correlation")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Mesh size $N_x$ / [a.u.]")
plt.ylabel("$\\ell^2$ norm, $||\\Delta c||_2$ / [a.u.]")
plt.ylim([5e-14, 5e3])

# plot lines for known orders of accuracy
hord = np.linspace(2 * gold_h, 50, 100)
Nord = np.array(L / hord)

plt.plot(Nord, log_hn(Nord, -1, np.log(4e3)), color="silver",
         label=r"$\mathcal{O}(h^1)$", zorder=0, linestyle="dotted")
plt.plot(Nord, log_hn(Nord, -2, np.log(6e3)), color="silver",
         label=r"$\mathcal{O}(h^2)$", zorder=0, linestyle="solid")
plt.plot(Nord, log_hn(Nord, -3, np.log(8e3)), color="silver",
         label=r"$\mathcal{O}(h^3)$", zorder=0, linestyle="dashed")
plt.plot(Nord, log_hn(Nord, -4, np.log(1e4)), color="silver",
         label=r"$\mathcal{O}(h^4)$", zorder=0, linestyle="dashdot")

# === Interpolate! ===

if not os.path.exists("../img"):
    os.mkdir("../img")

if not os.path.exists(f"{goldir}/interp"):
    os.mkdir(f"{goldir}/interp")

if not os.path.exists(f"{goldir}/img"):
    os.mkdir(f"{goldir}/img")

odir = f"interp_dx{gold_h:08.06f}"
if not os.path.exists(odir):
    os.mkdir(odir)

jobs = {}

for job in dirs:
    stats = sim_details(job)
    if stats["dx"] > gold_h:
        jobs[job] = stats

gold_npzs = sorted(glob.glob(f"{goldir}/c_????????.npz"))

for golden in gold_npzs:
    t = parse_npz.parse(golden)["t"]
    kolden = f"{goldir}/interp/k_{t:08d}.npz"
    gold_png = f"{goldir}/img/c_{t:08d}.png"
    ell_csv = f"{odir}/l2_{t:08d}.csv"

    if os.path.exists(ell_csv):
        df = pd.read_csv(ell_csv, index_col=0)
    else:
        df = None

    gold_c = upsampled(golden, kolden, gold_h, mesh_h, sinterp)

    if gold_c is not None:
        print(f"  Interpolating {variant.capitalize()}s @ t = {t:,d} / {gold_T:,d}")
        dataviz(kolden, gold_h, t, gold_png)

        for jobdir, job_par in jobs.items():
            startNorm = time.time()
            print(f"    {jobdir}:", end=" ")

            trpdir = f"{jobdir}/interp"
            if not os.path.exists(trpdir):
                os.mkdir(trpdir)

            imgdir = f"{jobdir}/img"
            if not os.path.exists(imgdir):
                os.mkdir(imgdir)

            job_N = job_par["Nx"]
            job_h = job_par["dx"]

            job_c_npz = f"{jobdir}/c_{t:08d}.npz"
            job_k_npz = f"{trpdir}/c_{t:08d}.npz"
            job_c_png = f"{imgdir}/c_{t:08d}.png"

            ell_two = None

            if df is not None and jobdir in df.index:
                ell_two = df.l2[jobdir]
            else:
                job_c = upsampled(job_c_npz, job_k_npz, job_h, mesh_h, sinterp)

                if job_c is not None:
                    ell_two = LA.norm(gold_c - job_c)

                    job_df = pd.DataFrame(
                        {
                            "h": job_h,
                            "N": job_N,
                            "l2": ell_two
                        },
                        index=[jobdir,]
                    )

                    df = pd.concat((df, job_df), axis=0, join='outer', sort=True)

            if ell_two is not None:
                dataviz(job_k_npz, job_h, t, job_c_png)
                watch = elapsed(startNorm)
                print(f" ℓ² = {ell_two:.04e}  ({watch:2d} s)")
            else:
                print(f"failed ({watch:2d} s).")

            gc.collect()

    plt.figure(1)
    plt.plot(df.N, df.l2, marker="o", label=f"$t={t:,d}$")

    df.to_csv(ell_csv, header=True, index=True, index_label="path")

if len(gold_npzs) < 50:
    plt.legend(ncol=3, loc=3, fontsize=9)

plt.savefig(png, dpi=400, bbox_inches="tight")

print(f"\n  Saved image to {png}\n")

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
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import os
from parse import parse
import sys
import time
# from zipfile import BadZipFile
try:
    from rich import print
except ImportError:
    print("Failed to import rich, going monochrome")

# import from `spectral.py` in same folder as the script
sys.path.append(os.path.dirname(__file__))
from spectral import FourierInterpolant as Interpolant


def elapsed(stopwatch):
    """
    Return the number of whole seconds elapsed since the mark
    """
    return np.ceil(time.time() - stopwatch).astype(int)


def sim_details(dir):
    _, dx = parse("dt{:6.04f}_dx{:08.04f}", dir)
    Nx = np.rint(200. / dx).astype(int)
    slices = sorted(glob.glob(f"{dir}/c_*.npz"))
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


class MidpointNormalize(matplotlib.colors.Normalize):
    """
    Helper class to center the colormap on a specific value from Joe Kington via
    <http://chris35wills.github.io/matplotlib_diverging_colorbar/>
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # ignoring masked values and lotsa edge cases
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


variant = os.path.basename(os.getcwd())

# reset color cycle for 20 lines
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.hsv(np.linspace(0, 1, 20)))

# parse command-line flags
parser = ArgumentParser()
parser.add_argument("--dx",   type=float, help="Candidate Gold Standard resolution")
parser.add_argument("--dt",   type=float, help="Timestep of interest")
parser.add_argument("--time", type=int,   help="Time slice(s) of interest, space-delimited",
                              default=0,  nargs="+")
args = parser.parse_args()

# get "gold standard" info
goldir = f"dt{args.dt:6.04f}_dx{args.dx:08.04f}"
gold_h, gold_N, gold_T = sim_details(goldir)

if gold_N % 2 != 0:
    raise ValueError("Reference mesh size is not even!")

print(f"=== {variant}/{goldir} has reached t={gold_T} ===")

# set output image file
png = f"norm_{variant}_dt{args.dt:6.04f}.png"

plt.figure(1, figsize=(10,8))
plt.title(f"IC: {variant}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Mesh size $N_x$ / [a.u.]")
plt.ylabel("$\\ell^2$ norm, $||\\Delta c||_2$ / [a.u.]")
plt.ylim([5e-14, 5e3])

# plot lines for known orders of accuracy
h = np.linspace(2 * gold_h, 50, 100)
N = 200 / h

plt.plot(N, log_hn(N, -1, np.log(4e3)), color="silver", label=r"$\mathcal{O}(h^1)$", zorder=0, linestyle="dotted")
plt.plot(N, log_hn(N, -2, np.log(6e3)), color="silver", label=r"$\mathcal{O}(h^2)$", zorder=0, linestyle="solid")
plt.plot(N, log_hn(N, -3, np.log(8e3)), color="silver", label=r"$\mathcal{O}(h^3)$", zorder=0, linestyle="dashdot")
plt.plot(N, log_hn(N, -4, np.log(1e4)), color="silver", label=r"$\mathcal{O}(h^4)$", zorder=0, linestyle="dashed")

# Interpolate!

sinterp = Interpolant((gold_N, gold_N))

jobs = {}

for job in sorted(glob.glob(f"dt{args.dt:6.04f}_dx???.????")):
    stats = sim_details(job)
    if stats[0] > gold_h:
        jobs[job] = stats


for golden in sorted(glob.glob(f"{goldir}/c_????????.npz")):
    resolutions = []
    norms = []

    _, t = parse("{}/c_{:d}.npz", golden)

    with np.load(golden) as npz:
        gold_c = npz["c"]

    print(f"\n  Interpolating {variant}s @ t = {t:,d} / {gold_T:,d}\n")

    for jobdir, (job_h, job_N, job_T) in jobs.items():
        print(f"    {jobdir}:", end=" ")
        terpdir = f"{jobdir}/interp"
        if not os.path.exists(terpdir):
            os.mkdir(terpdir)

        refined = f"{terpdir}/k_{t:08d}_h{gold_h:6.04f}.npz"
        refined_png = refined.replace("npz", "png")

        job_refined = None
        ell_two = None
        watch = None

        if not os.path.exists(refined):
            try:
                with np.load(f"{jobdir}/c_{t:08d}.npz") as npz:
                    job_c = npz["c"]

                startNorm = time.time()
                job_refined = sinterp.upsample(job_c)
                ell_two = LA.norm(gold_c - job_refined)
                np.savez_compressed(refined, c=job_refined, l2=ell_two)
                watch = elapsed(startNorm)
            except FileNotFoundError:
                job_refined = None
                ell_two = None


        with np.load(refined) as npz:
            ell_two = npz["l2"]

            if ell_two is not None:
                resolutions.append(job_N)
                norms.append(ell_two)
                if watch is not None:
                    print(f"ℓ² = {ell_two:.02e}  ({watch:2d} s)")
                else:
                    print(f"ℓ² = {ell_two:.02e}")
            else:
                print("failed.")

            if not os.path.exists(refined_png):
                job_refined = npz["c"]
                fig, axs = plt.subplots(1, 2, figsize=(10, 4),
                                        constrained_layout=True, sharex=True, sharey=True)

                fig.suptitle(f"$\\Delta x={job_h},\\ \\Delta t={args.dt}\\ @\\ t={t:,d}$")
                axs[0].set_xlabel("$x$ / [a.u.]")
                axs[0].set_ylabel("$y$ / [a.u.]")

                c_bulk = job_refined[1:-2,1:-2]

                c_min = np.amin(c_bulk)
                c_avg = np.average(c_bulk)
                c_max = np.amax(c_bulk)

                axs[0].set_title(r"$c$")
                fig.colorbar(
                    axs[0].imshow(job_refined, cmap="coolwarm", clim=(c_min, c_max),
                                  norm=MidpointNormalize(midpoint=c_avg, vmin=c_min, vmax=c_max),
                                  interpolation=None, origin="lower")
                )

                diff_c = np.absolute(job_refined - gold_c)
                axs[1].set_title(r"$(\Delta c)^2$")

                fig.colorbar(
                    axs[1].imshow(diff_c, norm="log",
                                  cmap="twilight_shifted", interpolation=None, origin="lower")
                )
                fig.savefig(refined_png, dpi=400, bbox_inches="tight")
                plt.close(fig)

        gc.collect()


    plt.figure(1)
    plt.plot(resolutions, norms, marker="o", label=f"$t={t:,d}$")


plt.legend(loc="best")
plt.savefig(png, dpi=400, bbox_inches="tight")
print(f"\n  Saved image to {png}\n")

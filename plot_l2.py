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
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import os
from parse import parse
import sys
import time
from zipfile import BadZipFile

# import from `spectral.py` in same folder as the script
sys.path.append(os.path.dirname(__file__))

from spectral import FourierInterpolant as Interpolant

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
    _, dx = parse("dt{:6.04f}_dx{:08.04f}", dir)
    Nx = np.rint(200. / dx).astype(int)
    slices = sorted(glob.glob(f"{dir}/c_*.npz"))
    _, t_max = parse("{}/c_{:08d}.npz", slices[-1])

    return float(dx), int(Nx), int(t_max)


def log_hn(h, n, b=0):
    """
    Support function for plotting 𝒪(hⁿ)
    h: array of dx values
    n: order of accuracy
    b: intercept
    """
    return b * h**n


# Helper class to center the colormap on a specific value
# from Joe Kington via <http://chris35wills.github.io/matplotlib_diverging_colorbar/>
class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# get "gold standard" info
goldir = f"dt{args.dt:6.04f}_dx{args.dx:08.04f}"
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

# plot lines for known orders of accuracy
h_ref = 2 * gold_h
h = np.linspace(h_ref, 6.25, 100)
N = 200 / h

plt.loglog(N, log_hn(N, -2, 1), color="silver", label=r"$\mathcal{O}(h^2)$", zorder=0)
plt.loglog(N, log_hn(N, -4, 1), color="silver", linestyle="dashed", label=r"$\mathcal{O}(h^4)$", zorder=0)

# Interpolate!

sinterp = Interpolant((gold_N, gold_N))

jobs = {}

for job in sorted(glob.glob(f"dt{args.dt:6.04f}_dx???.????")):
    stats = sim_details(job)
    if stats[0] > gold_h:
        jobs[job] = stats

times = np.unique(np.concatenate([np.array([0], dtype=int),
                                  np.array(args.time, dtype=int)]))
times = times[times <= gold_T]

ylim = np.array([1e6, 1e-6])

for t in times:
    resolutions = []
    norms = []

    with np.load(f"{goldir}/c_{t:08d}.npz") as npz:
        gold_c = npz["c"]

    for jobdir, (job_h, job_N, job_T) in jobs.items():
        print(f"Interpolating {variant}/{jobdir} @ t={t:,d}")
        terpdir = f"{jobdir}/interp"
        if not os.path.exists(terpdir):
            os.mkdir(terpdir)

        refined = f"{terpdir}/k_{t:08d}_h{gold_h:6.04f}.npz"
        job_refined = None

        if not os.path.exists(refined):
            try:
                with np.load(f"{jobdir}/c_{t:08d}.npz") as npz:
                    job_c = npz["c"]

                job_refined = sinterp.upsample(job_c)
                np.savez_compressed(refined, c=job_refined)
            except FileNotFoundError:
                job_refined = None
        else:
            try:
                with np.load(refined) as npz:
                    job_refined = npz["c"]
            except BadZipFile:
                job_refined = None

        if job_refined is not None:
            print("    L2: ", end="")
            ell_two = LA.norm(gold_c - job_refined)
            print(f"{ell_two:.02e}")

            resolutions.append(job_N)
            norms.append(ell_two)

            if ell_two > ylim[1]:
                ylim[1] = ell_two
            if ell_two < ylim[0]:
                ylim[0] = ell_two

            refined_png = refined.replace("npz", "png")
            if not os.path.exists(refined_png):
                fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True, sharex=True, sharey=True)
                fig.suptitle(f"$\\Delta x={job_h},\\ \\Delta t={args.dt}\\ @\\ t={t:,d}$")

                axs[0].set_xlabel("$x$ / [a.u.]")
                axs[0].set_ylabel("$y$ / [a.u.]")

                dmin = np.amin(job_refined[1:-2,1:-2])
                davg = 0.5
                dmax = np.amax(job_refined[1:-2,1:-2])

                axs[0].set_title(r"$c$")
                fig.colorbar(axs[0].imshow(job_refined, cmap="coolwarm", clim=(dmin, dmax),
                                           norm=MidpointNormalize(midpoint=davg, vmin=dmin, vmax=dmax),
                                           interpolation=None, origin="lower"))

                # fft_refined = np.fft.fftshift(np.fft.fft2(job_refined))
                # plt.colorbar(plt.imshow(fft_refined.real, norm="asinh", cmap="gray",
                #                         interpolation=None, origin="lower"))

                diff_c = np.absolute(job_refined - gold_c)
                axs[1].set_title(r"$(\Delta c)^2$")

                fig.colorbar(axs[1].imshow(diff_c, norm="log",
                                           cmap="twilight_shifted", interpolation=None, origin="lower"))
                fig.savefig(refined_png, dpi=400, bbox_inches="tight")
                plt.close(fig)

        gc.collect()

    print(f"Saving {variant} image to {png}")

    plt.figure(1)
    plt.loglog(resolutions, norms, marker="o", label=f"$t={t:,d}$")

    plt.ylim(ylim)
    plt.legend(loc="best")

    plt.savefig(png, dpi=400, bbox_inches="tight")

    print()

print(f"Saved image to {png}")

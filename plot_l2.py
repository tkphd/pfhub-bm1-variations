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
from spectral import generate_hash_table, interpolate

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

plt.ylim([1, 50])

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
        gold_c0 = npz["c"]

    for jobdir, (job_h, job_N, job_T) in jobs.items():
        print(f"Interpolating {jobdir} @ t={t:,d}")

        refined = f"{jobdir}/dx{gold_h:6.04f}_c_{t:08d}.npz"
        prehash = f"{goldir}/hash_{gold_N}_{job_N}.npz"

        table = None
        job_refined = None

        if not os.path.exists(refined):
            print("    Hashing: ", end="")

            if not os.path.exists(refined):
                try:
                    N_ratio = gold_N / job_N

                    if not N_ratio.is_integer():
                        raise ValueError("Mesh sizes are mismatched!")
                    elif job_N % 2 != 0:
                        raise ValueError("Mesh size is not even!")
                    else:
                        watch = time.time()
                        table = generate_hash_table(gold_N, job_N)
                        print(elapsed(watch), "s")

                        if np.all(np.isfinite(table)):
                            np.savez_compressed(prehash, table=table)
                        else:
                            table = None
                            raise ValueError("Collision in hash table!")
                except ValueError as e:
                    print(e)
            else:
                with np.load(prehash) as npz:
                    table = npz["table"]

            try:
                if table is not None:
                    print("    Intrpng: ", end="")
                    job_refined = np.zeros((gold_N, gold_N), dtype=float)

                    with np.load(f"{jobdir}/c_{t:08d}.npz") as npz:
                        job_c0 = npz["c"]

                    watch = time.time()
                    interpolate(job_c0, job_refined, table)
                    print(elapsed(watch), "s")

                    np.savez_compressed(refined, c=job_refined)
            except FileNotFoundError:
                job_refined = None
        else:
            with np.load(refined) as npz:
                job_refined = npz["c"]

        if job_refined is not None:
            print("    L2: ", end="")
            # L2 of zeroth step
            resolution.append(job_N)
            norm.append(gold_h * LA.norm(gold_c0 - job_refined))
            print(f"{norm[-1]:9.03e}")

        gc.collect()

    plt.loglog(resolution, norm, marker="o", label=f"$t={t:,d}$")

    print()

plt.legend(loc="best")
plt.savefig(png, dpi=400, bbox_inches="tight")

print(f"Saved image to {png}")

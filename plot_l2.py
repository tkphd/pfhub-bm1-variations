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

sys.path.append(os.path.dirname(__file__))

from spectral import generate_hash_table, interpolate

def elapsed(stopwatch):
    return np.round(time.time() - stopwatch, 2)

def sim_details(dir):
    _, dx = parse("dt{:6.4f}_dx{:6.4f}", dir)
    Nx = np.rint(200. / dx).astype(int)
    slices = sorted(glob.glob(f"{dir}/c_*.npz"))
    _, t_max = parse("{}/c_{}.npz", slices[-1])

    return dx, Nx, t_max

parser = ArgumentParser()
parser.add_argument("-t", "--dt", help="Timestep of interest", type=float)
args = parser.parse_args()

t = 0
Lk = 2 * np.pi

plt.figure(1, figsize=(10,8))
plt.title(f"$\Delta t = {args.dt}$")
plt.xlabel("Mesh size $N_x$ / [a.u.]")
plt.ylabel("L2 norm $||\\Delta c||_2$ / [a.u.]")

resolution = []
norm = []

dirs = sorted(glob.glob(f"dt{args.dt:6.04f}_dx?.????"))

# get "gold standard" info
goldir = dirs[0]  # smallest dx comes first

gold_h, gold_N, gold_T = sim_details(goldir)
hf = Lk / gold_N

with np.load(f"{goldir}/c_{t:08d}.npz") as npz:
    gold_c0 = npz["c"]

for jobdir in np.flip(np.array(dirs[1:])):
    print(f"Interpolating {jobdir}")

    job_h, job_N, job_T = sim_details(jobdir)
    refined = f"{jobdir}/dx{gold_h:6.04f}_c_{t:08d}.npz"
    prehash = f"{goldir}/hash_{gold_N}_{job_N}.npz"
    table = None
    job_refined = None

    if not os.path.exists(refined):
        print("    Hashing: ", end="")

        if not os.path.exists(refined):
            try:
                if not (gold_N / job_N).is_integer():
                    raise ValueError("Mesh sizes are mismatched!")
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

        if table is not None:
            print("    Intrpng: ", end="")
            fine_mesh = np.zeros((gold_N, gold_N), dtype=float)

            with np.load(f"{jobdir}/c_{t:08d}.npz") as npz:
                job_c0 = npz["c"]

            watch = time.time()
            job_refined = interpolate(job_c0, fine_mesh, table)
            print(elapsed(watch), "s")

            np.savez_compressed(refined, c=job_refined)
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


png = f"norm_dt{args.dt:6.04f}_{t:08d}.png"
plt.loglog(resolution, norm, marker="o")
plt.savefig(png, dpi=400, bbox_inches="tight")

print(f"Saved image to {png}")

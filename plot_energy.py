#!/usr/bin/env python3

import gc
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from parse import parse
from tqdm import tqdm

# load community submissions of note

subs = {}

for result in sorted(glob.glob("../1a_*_*.csv")):
    label = str(result).replace("../1a_", "").replace(".csv", "")
    subs[label] = pd.read_csv(result)

# load and plot spectral data

jobs = {}

dirs = sorted(glob.glob("dt?.????_dx?.????"))

for iodir in dirs:
    dt, dx = parse("dt{}_dx{}", iodir)
    dt = str(dt)
    if dt in jobs.keys():
        jobs[dt].append(iodir)
    else:
        jobs[dt] = [iodir]

fig_cols = 2
fig_rows = 2

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(4*fig_cols+1, 4*fig_rows),
                        constrained_layout=True)
axs = axs.flatten()

for idx, (dt, dirs) in enumerate(jobs.items()):
    print("")
    ax = axs[idx]
    ax.set_title(f"$\Delta t = {dt}$")

    ax.set_xlabel("Time $t$ / [a.u.]")
    ax.set_ylabel("Free energy $\\mathcal{F}$ / [a.u.]")
    ax.set_xlim([1.0, 2e5])

    # plot community uploads of note
    for label, df in subs.items():
        ax.loglog(df["time"], df["free_energy"], color="silver")

    for zord, iodir in enumerate(dirs):
        _, dx = parse("dt{:6.4f}_dx{:6.4f}", iodir)

        ene = f"{iodir}/ene.csv"

        df = pd.read_csv(ene)
        label = f"$\\Delta x = {dx:5.03f}$"
        ax.loglog(df["time"], df["free_energy"], label=label, zorder=10.0-zord)

        pbar = tqdm(sorted(glob.glob(f"{iodir}/c_*.npz")))
        for npz in pbar:
            pbar.set_description(iodir)
            img = npz.replace("npz", "png")

            if not os.path.exists(img):
                c = np.load(npz)
                if np.all(np.isfinite(c["c"])):
                    _, _, t = parse("dt{}_dx{}/c_{}.npz", npz)
                    t = int(t)

                    plt.figure(2, figsize=(10, 8))
                    plt.title(f"$\\Delta x={dx},\\ \\Delta t={dt}\\ @\\ t={t:,d}$")
                    plt.xlabel("$x$ / [a.u.]")
                    plt.ylabel("$y$ / [a.u.]")
                    plt.colorbar(plt.imshow(c["c"], interpolation=None, origin="lower"))
                    plt.savefig(img, dpi=400, bbox_inches="tight")

                    plt.close()

        gc.collect()

    ax.legend(loc="best", fontsize=8)

# Render to PNG

plt.savefig("energy.png", dpi=400, bbox_inches="tight")

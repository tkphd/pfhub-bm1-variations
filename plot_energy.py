#!/usr/bin/env python3

import gc
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from parse import parse
from tqdm import tqdm
from zipfile import BadZipFile

variant = os.path.basename(os.getcwd())

# reset color cycle for full range of datasets
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.rainbow(np.linspace(0, 1, 20)))

# load community submissions of note

subs = {}

for result in sorted(glob.glob("../pfhub/1a_*_*.csv")):
    label = str(result).replace("../pfhub/1a_", "").replace(".csv", "")
    subs[label] = pd.read_csv(result)

# survey spectral data

jobs = {}

dirs = sorted(glob.glob("dt?.????_dx???.????"))

for iodir in dirs:
    dt, dx = parse("dt{}_dx{}", iodir)
    dt = str(dt)
    if dt in jobs.keys():
        jobs[dt].append(iodir)
    else:
        jobs[dt] = [iodir]

fig_cols = 2
fig_rows = 1
figsize = (4 * fig_cols+1, 4 * fig_rows)

fig, axs = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                        constrained_layout=True)
axs = axs.flatten()

xlim = (1, 1.5e6)
ylim = (10, 350)

# plot community results

ax = axs[0]

ax.set_title("original IC: community uploads")
ax.set_xlabel("Time $t$ / [a.u.]")
ax.set_ylabel("Free energy $\\mathcal{F}$ / [a.u.]")
ax.set_xlim(xlim)
ax.set_ylim(ylim)

for label, df in subs.items():
    ax.loglog(df["time"], df["free_energy"], label=label)

ax.legend(loc=3, fontsize=6)

# plot spectral results

ax = axs[1]
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_title(f"$\Delta t = {dt}$")
ax.set_xlabel("Time $t$ / [a.u.]")
ax.set_ylabel("Free energy $\\mathcal{F}$ / [a.u.]")

ax.set_xscale("log")
ax.set_yscale("log")

for dt, dirs in jobs.items():
    print("")

    # plot community uploads of note
    for label, df in subs.items():
        ax.plot(df["time"], df["free_energy"], color="silver", zorder=0.0)

    for zord, iodir in enumerate(dirs):
        priority = 10 - 9 * zord / len(dirs)
        _, dx = parse("dt{:6.4f}_dx{:6.4f}", iodir)

        ene = f"{iodir}/ene.csv"

        df = pd.read_csv(ene)
        label = f"{variant} IC: $\\Delta x = {dx}$"
        ax.plot(df["time"], df["free_energy"], label=label, zorder=priority)

        if np.isclose(dx, 0.0625):
            ax.scatter(df["time"].iloc[-1], df["free_energy"].iloc[-1],
                       s=2, color="black", zorder=priority)

        pbar = tqdm(sorted(glob.glob(f"{iodir}/c_*.npz")))
        for npz in pbar:
            pbar.set_description(iodir)
            img = npz.replace("npz", "png")

            if not os.path.exists(img):
                try:
                    c = np.load(npz)
                    if np.all(np.isfinite(c["c"])):
                        _, _, t = parse("dt{}_dx{}/c_{}.npz", npz)
                        t = int(t)

                        plt.figure(2, figsize=(10, 8))
                        plt.title(f"{variant} IC: $\\Delta x={dx},\\ \\Delta t={dt}\\ @\\ t={t:,d}$")
                        plt.xlabel("$x$ / [a.u.]")
                        plt.ylabel("$y$ / [a.u.]")
                        plt.colorbar(plt.imshow(c["c"], interpolation=None, origin="lower"))
                        plt.savefig(img, dpi=400, bbox_inches="tight")

                        plt.close()
                except BadZipFile:
                    pass

        gc.collect()

    ax.legend(ncol=2, loc=3, fontsize=6)

# Render to PNG

plt.savefig("energy.png", dpi=400, bbox_inches="tight")

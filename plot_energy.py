#!/usr/bin/env python3

import gc
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from parse import parse
from tqdm import tqdm
from zipfile import BadZipFile

mpl.use("Agg")
variant = os.path.basename(os.getcwd())

# reset color cycle for full range of datasets
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.rainbow(np.linspace(0, 1, 20)))

# survey spectral data

jobs = {}
dirs = sorted(glob.glob("*dx???.????"))

for iodir in dirs:
    if iodir.__contains__("dt"):
        dt, dx = parse("dt{}_dx{}", iodir)
    else:
        dt = "0.0000"
        _, dx = parse("{}{}", iodir)
    dt = str(dt)
    if dt in jobs.keys():
        jobs[dt].append(iodir)
    else:
        jobs[dt] = [iodir]


x_lim = (1, 1e6)
y_lim = (10, 350)

for dt, dirs in jobs.items():

    # prepare free energy plot

    plt.figure(1, figsize=(10,8))
    plt.title(f"\"{variant.capitalize()}\" IC: Free Energy")

    plt.xlabel("Time $t$ / [a.u.]")
    plt.xlim(x_lim)
    plt.xscale("log")

    plt.ylabel("Free energy $\\mathcal{F}$ / [a.u.]")
    plt.ylim(y_lim)
    plt.yscale("log")

    # prepare timestep plot

    plt.figure(2, figsize=(10,8))
    plt.title(f"\"{variant.capitalize()}\" IC: Timestep")

    plt.xlabel("Time $t$ / [a.u.]")
    plt.xlim(x_lim)

    plt.ylabel("Timestep $\\Delta t$ / [a.u.]")
    plt.yscale("log")

    print("")

    for zord, iodir in enumerate(dirs):
        ene = f"{iodir}/ene.csv"
        priority = 10 - 9 * zord / len(dirs)
        _, dx = parse("{}{:08.04f}", iodir)
        label = f"$\\Delta x = {dx}$"

        df = pd.read_csv(ene)
        df.drop_duplicates(subset=None, inplace=True)

        plt.figure(1)
        plt.plot(df["time"], df["free_energy"], label=label, zorder=priority)

        # indicate current time of the gold standard
        if np.isclose(dx, 0.0625):
            plt.plot(
                (df["time"].iloc[-1], df["time"].iloc[-1]),
                (df["free_energy"].iloc[-1], y_lim[0]),
                color="silver", linestyle="dotted", zorder=priority
            )

        plt.figure(2)
        plt.plot(df["time"][1:], np.diff(df["time"]), label=label, zorder=priority)

        pbar = tqdm(sorted(glob.glob(f"{iodir}/c_*.npz")))
        for npz in pbar:
            pbar.set_description(iodir)
            img = npz.replace("npz", "png")

            if not os.path.exists(img):
                try:
                    c = np.load(npz)
                    if np.all(np.isfinite(c["c"])):
                        _, t = parse("{}/c_{}.npz", npz)
                        t = int(t)

                        plt.figure(3, figsize=(10, 8))
                        plt.title(f"\"{variant.capitalize()}\" IC: $\\Delta x={dx}\\ @\\ t={t:,d}$")
                        plt.xlabel("$x$ / [a.u.]")
                        plt.ylabel("$y$ / [a.u.]")
                        plt.colorbar(
                            plt.imshow(c["c"], interpolation=None, origin="lower")
                        )
                        plt.savefig(img, dpi=400, bbox_inches="tight")
                        plt.close()
                except BadZipFile:
                    pass

        gc.collect()

    # Render to PNG
    plt.figure(1)
    plt.legend(ncol=2, loc=3, fontsize=6)
    plt.savefig(f"energy_{variant}_dt{dt}.png", dpi=400, bbox_inches="tight")
    plt.close()

    plt.figure(2)
    plt.legend(ncol=2, loc=3, fontsize=6)
    plt.savefig(f"timestep_{variant}_dt{dt}.png", dpi=400, bbox_inches="tight")
    plt.close()

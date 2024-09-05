#!/usr/bin/env python3

import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from parse import compile

mpl.use("Agg")
variant = os.path.basename(os.getcwd())

# reset color cycle for full range of datasets
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.rainbow(np.linspace(0, 1, 20)))

# my goofy folder naming conventions
job_pattern = "dx???.????"
parse_job = compile("dx{dx:8f}")
parse_npz = compile("{}/c_{}.npz")
dirs = sorted(glob.glob(job_pattern))

# survey spectral data

jobs = {}

for iodir in dirs:
    deets = parse_job.parse(iodir)
    if deets is not None:
        dx = deets["dx"]
        dt = 1

    if dt in jobs.keys():
        jobs[dt].append(iodir)
    else:
        jobs[dt] = [iodir]

# x_lim = (0.0625, 2e6)
# y_lim = (10, 350)

for dt, dirs in jobs.items():

    # prepare free energy plot

    plt.figure(1, figsize=(9,7.2))
    plt.title(f"\"{variant.capitalize()}\" IC: Free Energy")

    plt.xlabel("Time $t$ / [a.u.]")
    # plt.xlim(x_lim)
    plt.xscale("log")

    plt.ylabel("Free energy $\\mathcal{F}$ / [a.u.]")
    # plt.ylim(y_lim)
    plt.yscale("log")

    for zord, iodir in enumerate(dirs):
        priority = 10 - 9 * zord / len(dirs)
        ene = f"{iodir}/ene.csv.gz"
        deets = parse_job.parse(iodir)
        if deets is None:
            deets = parse_job.parse(iodir)

        dx = deets["dx"]

        label = f"$\\Delta x = {dx}$"

        try:
            df = pd.read_csv(ene)

            plt.figure(1)

            plt.plot(df["time"], df["free_energy"], label=label, zorder=priority)

            # indicate current time of the gold standard
            if np.isclose(float(dx), 0.0625):
                plt.plot(
                    (df["time"].iloc[-1], df["time"].iloc[-1]),
                    (df["free_energy"].iloc[-1], 10),
                    color="silver", linestyle="dotted", zorder=priority
                )

        except FileNotFoundError:
            df = None
            pass

    # Render to PNG
    plt.figure(1)
    plt.legend(ncol=2, loc=3, fontsize=6)
    plt.savefig(f"energy_{variant}_dt{dt}.png", dpi=400, bbox_inches="tight")
    plt.close()

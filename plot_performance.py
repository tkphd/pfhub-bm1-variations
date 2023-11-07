#!/usr/bin/env python3

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import parse

# identify gold standard variants

goldirs = list(sorted(glob.glob("*/*dx000.0625", recursive=True)))

# reset color cycle for full range of datasets
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.rainbow(np.linspace(0, 1, len(goldirs))))

# survey spectral data

jobs = {}

for iodir in goldirs:
    variant, _ = parse.parse("{}/{}000.0625", str(iodir))
    jobs[variant] = os.path.join(iodir, "ene.csv")

# plot the runtime performance

plt.figure(1, figsize=(10, 8))
plt.title("IC Performance")
plt.xlabel("Wall Time / [s]")
plt.ylabel("Fictive Time / [a.u.]")

plt.figure(2, figsize=(10, 8))
plt.title("IC Residue")
plt.xlabel("Wall Time / [s]")
plt.ylabel("Residual / [a.u.]")
plt.yscale("log")

res_plot = False

for variant, ene in jobs.items():
    df = pd.read_csv(ene)
    df.drop_duplicates(subset=None, inplace=True)
    label = f"{variant.capitalize()} IC"

    plt.figure(1)
    plt.plot(df["runtime"], df["time"], label=label)

    if "residual" in df.columns:
        res_plot = True
        plt.figure(2)
        plt.plot(df["runtime"], df["residual"], label=label)

# render to PNG

plt.figure(1)
plt.legend(ncol=2, loc="best", fontsize=6)
plt.savefig("performance.png", dpi=400, bbox_inches="tight")
plt.close()

plt.figure(2)
if res_plot:
    plt.legend(ncol=2, loc="best", fontsize=6)
    plt.savefig("residual.png", dpi=400, bbox_inches="tight")
plt.close()

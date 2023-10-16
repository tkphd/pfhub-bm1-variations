#!/usr/bin/env python3

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import parse

# identify gold standard variants

goldirs = list(sorted(glob.glob("*/dt0.1250_dx000.0625", recursive=True)))

# reset color cycle for full range of datasets
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.rainbow(np.linspace(0, 1, len(goldirs))))

# survey spectral data

jobs = {}

for iodir in goldirs:
    variant = parse.parse("{}/dt0.1250_dx000.0625", str(iodir))[0]
    jobs[variant] = os.path.join(iodir, "ene.csv")

# plot the runtime performance

plt.figure(1, figsize=(10, 8))
plt.title("IC Performance")
plt.xlabel("Wall Time / [s]")
plt.ylabel("Fictive Time / [a.u.]")

for variant, ene in jobs.items():
    df = pd.read_csv(ene)
    label = f"{variant.capitalize()} IC"
    plt.plot(df["runtime"], df["time"], label=label)

# render to PNG

plt.legend(ncol=2, loc="best", fontsize=6)

plt.savefig("performance.png", dpi=400, bbox_inches="tight")

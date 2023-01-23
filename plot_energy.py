#!/Usr/bin/env python
# coding: utf-8

# Plot free energy and memory consumption

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import pandas
from os import path

from argparse import ArgumentParser

variants = ("orig", "peri", "zany")
figsize = (6, 5)
dpi=400

parser = ArgumentParser(
    prog = 'fipy-bm1-variations',
    description = 'Plot results of PFHub BM1 in FiPy with periodic initial condition variations'
)

args = parser.parse_args()

def read_and_plot(iodir):
    csv_file = "{}/energy.csv".format(iodir)
    df = None
    if path.exists(csv_file):
        df = pandas.read_csv("{}/energy.csv".format(iodir))
        df.time[0] = 0.875 * df.time[1]

        plt.figure(1, figsize=figsize)
        plt.xlabel(r"time $t$ / [a.u.]")
        plt.ylabel(r"Free energy $\mathcal{F}$ / [J/m³]")
        plt.loglog(df.time, df.free_energy, label=iodir)
        plt.savefig("{}/energy.png".format(iodir), bbox_inches="tight", dpi=dpi)
        plt.close()

        plt.figure(2, figsize=figsize)
        plt.xlabel(r"time $t$ / [a.u.]")
        plt.ylabel("Memory / [GB]")
        plt.loglog(df.time, df.mem_GB)
        plt.savefig("{}/memory.png".format(iodir), bbox_inches="tight", dpi=dpi)
        plt.close()

    return df


def plot_all(dirs):
    dframes = {}

    # create energy & memory plots for each initial condition
    for dirname in dirs:
        if path.exists(dirname):
            dframes[dirname] = read_and_plot(dirname)

    # co-plot all energy results
    plt.figure(3, figsize=figsize)
    plt.xlabel(r"time $t$ / [a.u.]")
    plt.ylabel(r"Free energy $\mathcal{F}$ / [J/m³]")
    for dirname, df in dframes.items():
        if df is not None:
            plt.loglog(df.time, df.free_energy, label=dirname)
    plt.legend(loc="best")
    plt.savefig("energies.png", bbox_inches="tight", dpi=dpi)
    plt.close()

if __name__ == "__main__":
    # read_and_plot(iodir)
    plot_all(variants)

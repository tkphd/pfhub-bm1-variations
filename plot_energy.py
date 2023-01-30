#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot free energy and memory consumption

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import pandas
from os import path

from argparse import ArgumentParser

variants = {
    "orig": "original",
    "peri": "periodic",
    "zany": "timbrous"
}

figsize = (6, 5)
ressize = (11, 6)
dpi=400

parser = ArgumentParser(
    prog = 'plot_energy',
    description = 'Plot results of PFHub BM1 in FiPy with periodic initial condition variations'
)

parser.add_argument("-p", "--prefix",
                    default=".",
                    help="prefix (directory) for variants")
parser.add_argument("-s", "--software",
                    default="FiPy",
                    help="software platform")

args = parser.parse_args()

marker_style = dict(marker='.',
                    linestyle='-',
                    linewidth=0.5,
                    markersize=1,
                    alpha=0.6)

res_style = dict(marker='.',
                 linewidth=0,
                 markersize=1,
                 alpha=1)

def read_and_plot(prefix, variant):
    iodir = prefix + "/" + variant
    data_file = "{}/energy.csv".format(iodir)
    if not path.exists(data_file):
        data_file = "{}/free_energy.csv.gz".format(iodir)

    if path.exists(data_file):
        df = pandas.read_csv(data_file)
        df.time[0] = 0.95 * df.time[1]  # no zeros in log-log plot

        mem_col = [c for c in df.columns.values if c.startswith("mem")]
        mem_scl = 1024 if "KB" in mem_col[0] else 1

        plt.figure(1)
        plt.loglog(df.time, df.free_energy, label=variants[variant])

        plt.figure(2)
        plt.loglog(df.time, df[mem_col] / mem_scl, label=variants[variant])

def residual_plot(prefix, variant, ax, rtol=1e-3):
    iodir = prefix + "/" + variant
    data_file = "{}/residual.csv.gz".format(iodir)
    tmax = 0.0

    if path.exists(data_file):
        df = pandas.read_csv(data_file)

        ax.set_xlabel(r"time $t$ / [a.u.]")
        ax.set_ylabel(r"Residual / $10^{-3}$")
        ax.set_ylim([2e-1, 3e2])

        tvals = df.time + df.timestep * (100 + df.sweep)/100

        ax.semilogy(tvals, df.residual / rtol,
                    label=variants[variant], fillstyle=None, **marker_style, zorder=1)

        ini = df[df.sweep == 0]
        fin = df[df.sweep == 4]

        ax.semilogy(ini.time,
                    ini.residual / rtol,
                    label="first",
                    **res_style, zorder=1)
        ax.semilogy(fin.time + fin.timestep * (100 + fin.sweep)/100,
                    fin.residual / rtol,
                    label="last", color="black",
                    **res_style, zorder=1)

        ax.legend(loc="lower left", ncol=3)

        tmax = max(tmax, tvals.max())

    return tmax

def plot_all(prefix=".", software="FiPy"):
    # create energy & memory plots for each variant,
    # and co-plot all energy results

    nrg_image = "{}/energy.png".format(prefix)
    mem_image = "{}/memory.png".format(prefix)
    res_image = "{}/residual.png".format(prefix)

    # prepare energy plot
    plt.figure(1, figsize=figsize)
    plt.title("BM 1a: {}".format(software))
    plt.xlabel(r"time $t$ / [a.u.]")
    plt.ylabel(r"Free energy $\mathcal{F}$ / [J/m³]")

    # prepare memory plot
    plt.figure(2, figsize=figsize)
    plt.title("BM 1a: {}".format(software))
    plt.xlabel(r"time $t$ / [a.u.]")
    plt.ylabel("Memory / [MB]")

    # plot the energy data
    for variant in variants.keys():
        read_and_plot(prefix, variant)

    # save the plots
    plt.figure(1)
    plt.legend(loc="best")
    plt.savefig(nrg_image, bbox_inches="tight", dpi=dpi)
    plt.close()

    plt.figure(2)
    plt.legend(loc="best")
    plt.savefig(mem_image, bbox_inches="tight", dpi=dpi)
    plt.close()

    # plot the residual data, if available
    haveResiduals = any([path.exists("{}/{}/residual.csv.gz".format(prefix, variant)) for variant in variants.keys()])
    if haveResiduals:
        fig, axs = plt.subplots(len(variants.keys()), 1, figsize=ressize)
        fig.suptitle("BM 1a: {}".format(software))
        tmax = 0.0

        for i, variant in enumerate(variants.keys()):
            tmax = max(tmax, residual_plot(prefix, variant, axs[i]))

        for ax in axs:
            ax.set_xlim([0, min(tmax, 10_001)])

        plt.savefig(res_image, bbox_inches="tight", dpi=dpi)
        plt.close()


if __name__ == "__main__":
    # read_and_plot(iodir)
    plot_all(args.prefix, args.software)

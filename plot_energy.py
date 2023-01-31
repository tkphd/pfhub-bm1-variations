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
dpi=600

parser = ArgumentParser(
    prog = 'plot_energy',
    description = 'Plot results of PFHub BM1 in FiPy with periodic initial condition variations'
)

parser.add_argument("-d", "--directory",
                    default=".",
                    help="simulation output directory")
parser.add_argument("-p", "--platform",
                    default="FiPy",
                    help="software platform")
parser.add_argument("-s", "--sweeps",
                    default=None,
                    type=int,
                    help="number of sweeps per solver step")

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

def read_and_plot(iodir, variant):
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

def residual_plot(ax, iodir, variant, sweeps, rtol=1e-3):
    data_file = "{}/residual.csv.gz".format(iodir)
    tmax = 0.0

    if path.exists(data_file):
        df = pandas.read_csv(data_file)

        ax.set_xlabel(r"time $t$ / [a.u.]")
        ax.set_ylabel(r"Res / Tol")
        ax.set_ylim([1e-1, 2e2])

        tvals = df.time + df.timestep * (100 + df.sweep)/100

        ax.semilogy(tvals, df.residual / rtol,
                    label=variants[variant], fillstyle=None, **marker_style, zorder=1)

        ini = df[df.sweep == 0]
        fin = df[df.sweep == sweeps - 1]

        ax.semilogy(ini.time,
                    ini.residual / rtol,
                    label="first",
                    **res_style, zorder=1)
        ax.semilogy(fin.time + fin.timestep * (100 + fin.sweep)/100,
                    fin.residual / rtol,
                    label="last", color="black",
                    **res_style, zorder=1)

        ax.legend(loc="upper right", ncol=3)

        tmax = max(tmax, tvals.max())

    return tmax

def plot_all(prefix=".", platform="FiPy", suffix=None):
    # create energy & memory plots for each variant,
    # and co-plot all energy results

    nrg_image = f"{prefix}/energy{suffix}.png"
    mem_image = f"{prefix}/memory{suffix}.png"
    res_image = f"{prefix}/residual{suffix}.png"

    # prepare energy plot
    plt.figure(1, figsize=figsize)
    plt.title("BM 1a: {}".format(platform))
    plt.xlabel(r"time $t$ / [a.u.]")
    plt.ylabel(r"Free energy $\mathcal{F}$ / [J/m³]")

    # prepare memory plot
    plt.figure(2, figsize=figsize)
    plt.title("BM 1a: {}".format(platform))
    plt.xlabel(r"time $t$ / [a.u.]")
    plt.ylabel("Memory / [MB]")

    # plot the energy data
    for variant in variants.keys():
        iodir = f"{prefix}/{variant}{suffix}"
        read_and_plot(iodir, variant)

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
    haveResiduals = any([path.exists(f"{prefix}/{variant}{suffix}/residual.csv.gz")
                         for variant in variants.keys()])
    if haveResiduals:
        fig, axs = plt.subplots(len(variants.keys()), 1, figsize=ressize)
        fig.suptitle("BM 1a: {}".format(platform))
        tmax = 0.0

        for i, variant in enumerate(variants.keys()):
            iodir = f"{prefix}/{variant}{suffix}"
            tmax = max(tmax,
                       residual_plot(axs[i], iodir, variant, args.sweeps))

        for ax in axs:
            # ax.set_xlim([0, min(tmax, 10_001)])
            ax.set_xlim([0, tmax])

        plt.savefig(res_image, bbox_inches="tight", dpi=dpi)
        plt.close()


if __name__ == "__main__":
    # read_and_plot(iodir)
    suffix = None if args.sweeps is None \
        else f"-{args.sweeps:02d}sw"
    plot_all(args.directory, args.platform, suffix)

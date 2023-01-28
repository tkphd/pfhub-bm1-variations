#!/Usr/bin/env python
# coding: utf-8

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

def read_and_plot(prefix, variant, software=None):
    iodir = prefix + "/" + variant
    data_file = "{}/energy.csv".format(iodir)
    if not path.exists(data_file):
        data_file = "{}/free_energy.csv.gz".format(iodir)
    nrg_image = "{}/energy.png".format(iodir)
    mem_image = "{}/memory.png".format(iodir)

    if path.exists(data_file):
        df = pandas.read_csv(data_file)
        df.time[0] = 0.95 * df.time[1]  # no zeros in log-log plot
        mem_col = [c for c in df.columns.values if c.startswith("mem")]
        mem_scl = mem_col[0].replace("mem_", "")

        plt.figure(1, figsize=figsize)
        if software is not None:
            plt.title("BM 1a: {}".format(software))
        plt.xlabel(r"time $t$ / [a.u.]")
        plt.ylabel(r"Free energy $\mathcal{F}$ / [J/m³]")
        plt.loglog(df.time, df.free_energy, label=variants[variant])
        plt.legend(loc="best")
        plt.savefig(nrg_image, bbox_inches="tight", dpi=dpi)
        plt.close()

        plt.figure(2, figsize=figsize)
        if software is not None:
            plt.title("BM 1a: {}".format(software))
        plt.xlabel(r"time $t$ / [a.u.]")
        plt.ylabel("Memory / [{}]".format(mem_scl))
        plt.loglog(df.time, df[mem_col], label=variants[variant])
        plt.legend(loc="best")
        plt.savefig(mem_image, bbox_inches="tight", dpi=dpi)
        plt.close()

        if plt.fignum_exists(3):
            plt.figure(3)
            plt.loglog(df.time, df.free_energy, label=variants[variant])


def residual_plot(prefix, variant, software=None):
    iodir = prefix + "/" + variant
    data_file = "{}/residual.csv.gz".format(iodir)
    res_image = "{}/residual.png".format(iodir)
    if path.exists(data_file):
        df = pandas.read_csv(data_file)
        plt.figure(1, figsize=figsize)
        if software is not None:
            plt.title("BM 1a: {}".format(software))
        plt.xlabel(r"time $t$ / [a.u.]")
        plt.ylabel(r"Residual")
        plt.scatter(df.time * (100 + df.sweep)/100, df.residual, alpha=0.5, label=variants[variant])
        plt.yscale("log")
        plt.legend(loc="best")
        plt.savefig(res_image, bbox_inches="tight", dpi=dpi)
        plt.close()


def plot_all(prefix=".", software="FiPy"):
    # create energy & memory plots for each variant,
    # and co-plot all energy results
    plt.figure(3, figsize=figsize)
    plt.title("BM 1a: {}".format(software))
    plt.xlabel(r"time $t$ / [a.u.]")
    plt.ylabel(r"Free energy $\mathcal{F}$ / [J/m³]")
    for variant in variants.keys():
        read_and_plot(prefix, variant, software)
        residual_plot(prefix, variant, software)
    plt.figure(3)
    plt.legend(loc="best")
    nrg_image = "{}/energies.png".format(prefix)
    plt.savefig(nrg_image, bbox_inches="tight", dpi=dpi)
    plt.close()

if __name__ == "__main__":
    # read_and_plot(iodir)
    plot_all(args.prefix, args.software)

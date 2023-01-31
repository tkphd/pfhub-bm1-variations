#!/usr/bin/env python3
# coding: utf-8

# Endpoint detection: volume-weighted time rate of change of the free energy
# (per <https://doi.org/10.1016/j.commatsci.2016.09.022>)
# (dF/dt) / V < 1e-14

## PFHub BM 1a in FiPy with Steppyngstounes
#
# This notebook implements [PFHub] Benchmark [1a][spinodal] using [FiPy] and [steppyngstounes].
# It also explores alternative initial conditions that are more-periodic than the specification.
#
# <!-- links -->
# [FiPy]: https://www.ctcms.nist.gov/fipy
# [spinodal]: https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/#(a)-Square-periodic
# [steppyngstounes]: https://github.com/usnistgov/steppyngstounes
# [PFHub]: https://pages.nist.gov/pfhub/

import time
startTime = time.time()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import psutil

from argparse import ArgumentParser

from fipy import numerix as np
from fipy import CellVariable
from fipy import DiffusionTerm, ImplicitSourceTerm, TransientTerm
from fipy import parallel, MultiViewer
from fipy import Matplotlib2DViewer as Viewer
from fipy import PeriodicGrid2D as Grid2D

from fipy.solvers.petsc import LinearGMRESSolver as Solver
from petsc4py import PETSc

try:
    from rich import print
except ImportError:
    pass

from steppyngstounes import CheckpointStepper, PIDStepper

from tqdm import tqdm

# ==============================================================

parser = ArgumentParser(
    prog = 'fipy-bm1-variations',
    description = 'PFHub BM1 in FiPy with periodic initial condition variations'
)

parser.add_argument("-p", "--prefix", default="fipy", type=str,
                    help="output directory name")
parser.add_argument("-s", "--sweeps",  default=5,      type=int,
                    help="number of sweeps per solver step")
parser.add_argument("-v", "--variant", default="orig", type=str,
                    help="one of 'orig', 'peri', or 'zany'")

args = parser.parse_args()
suffix = "{:02d}sw".format(args.sweeps)

rank = parallel.procID
rankIsHead = (rank == 0)

def mprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


if not rankIsHead:
    raise RuntimeError("ParallelGrid2D does not support parallel execution.")

if not os.path.exists(args.prefix):
    os.mkdir(args.prefix)

iodir = "{}/{}-{}".format(args.prefix, args.variant, suffix)
if not os.path.exists(iodir):
    os.mkdir(iodir)
mprint(f"Writing simulation output to {iodir}")

ceil = np.ceil
cos  = np.cos
exp  = np.exp
log  = np.log
pi   = np.pi

proc = psutil.Process()
mpl.use("agg")

### Prepare mesh & field variables

Lx = Ly = 200
dx = dy = 1.0  # 200×200

nx = int(Lx / dx)
ny = int(Ly / dy)

mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
x, y = mesh.cellCenters

c = CellVariable(mesh=mesh, name=r"$c$",   hasOld=True)
μ = CellVariable(mesh=mesh, name=r"$\mu$", hasOld=True)

#### Set thermo-kinetic constants from the BM1 specification

α = 0.3
β = 0.7
ρ = 5.0
κ = 2.0
M = 5.0
ζ = 0.5  # mean composition
ϵ = 0.01 # noise amplitude

t = 0.0
dt = 1e-5
rtol = 1e-3

t_fin = 20_000  # nothing should take this long
t_min = 10_000  # nothing should end before this
f_fin = 1e-16   # final rate of free energy evolution

### Prepare free energy output

nrg_file = "{}/energy.csv".format(iodir) if rankIsHead else None
res_file = "{}/residual.csv.gz".format(iodir) if rankIsHead else None
chk_file = "{}/checkpoint.npz".format(iodir) if rankIsHead else None

nrg_cols = (
    "wall_time",
    "time",
    "free_energy",
    "mem_MB",
    "timestep",
    "mass",
    "energy_rate"
)

res_cols = (
    "time",
    "timestep",
    "sweep",
    "residual",
    "success"
)

# Write to disk uniformly in logarithmic space
checkpoints = np.unique(
    [
        int(float(exp(q))) for q in
        np.arange(0, log(t_fin) + 0.1, 0.1)
    ]
)

restartFromCheckpoint = os.path.exists(chk_file)

# === Utility Functions ===

def dump_restart(fnpz=None):
    np.savez_compressed(fnpz, c=c.value, u=μ.value, t=t, dt=dt)

def load_restart(fnpz=None):
    global t
    global c
    global μ

    with np.load(fnpz) as data:
        c.value[:] = data["c"]
        μ.value[:] = data["u"]
        t = data["t"]

def fbulk(C):
    return ρ * (C - α)**2 * (β - C)**2

def energy_rate(df=None):
    if df is None or len(df) < 2:
        return 1.0
    V = Lx * Ly
    delF = float(df.free_energy.iloc[-1] - df.free_energy.iloc[-2])

    return -delF / (V * dt)

def update_energy(df=None):
    # Integration of fields
    nrg = float((fbulk(c) + 0.5 * κ * (c.grad.mag)**2).sum())
    mas = c.sum()
    mem = parallel.allgather(proc.memory_info().rss) / 1024**2
    if rankIsHead:
        dFv_dt = float(energy_rate(df))
        timer = time.time() - startTime
        vals = (timer, t, nrg, mem, dt, mas, dFv_dt)

        index = [0] if (df is None) else [len(df)]
        update = pd.DataFrame([vals], columns=nrg_cols, index=index)

        return pd.concat([df, update])

    return None

def write_dataframe(filename=None, dataframe=None):
    if rankIsHead and dataframe is not None and filename is not None:
        if filename.endswith(".gz"):
            compression = "gzip"
        else:
            compression = None
        dataframe.to_csv(filename, index=False, compression=compression)

def write_plot(fig, viewers):
    t_int = int(t)
    imgname = f"{iodir}/bm1a.{t_int:09_d}.png"
    if rankIsHead and not os.path.exists(imgname):
        fig.suptitle(f"$t = {t:,f}$")
        viewers.plot()
        plt.savefig(imgname, bbox_inches="tight", dpi=400)

### Define equations of motion
#
# This is based on [fipy.examples.cahnHilliard.mesh2DCoupled],
# using a first-order Taylor series substitution in place of the bulk free energy "source term".
# The coupled equations are
#
# $$
# \frac{\partial c}{\partial t} = \nabla \cdot M\nabla\mu
# $$
#
# $$
# \mu = \left(\frac{\partial f_{\mathrm{bulk}}}{\partial c} - \frac{\partial^2 f_{\mathrm{bulk}}}{\partial c^2} \cdot c\right)_{\mathrm{old}} + \frac{\partial^2 f_{\mathrm{bulk}}}{\partial c^2}\cdot c - \kappa \nabla^2 c
# $$
#
# where the second term on $\mu$ is an `ImplicitSourceTerm` and the last is a `DiffusionTerm`.
#
# [fipy.examples.cahnHilliard.mesh2DCoupled]: https://www.ctcms.nist.gov/fipy/examples/cahnHilliard/generated/examples.cahnHilliard.mesh2DCoupled.html

# The free energy density and its first two derivatives are (refactored after SymPy)
#
# $$f_{\mathrm{bulk}} = \rho (c - \alpha)^2 (\beta - c)^2$$
#
# $$f'_{\mathrm{bulk}} = 2\rho (c - \alpha)(\beta - c)(\alpha - 2c + \beta)$$
#
# $$f''_{\mathrm{bulk}} = 2\rho\left\{\alpha^2 + 4 \alpha \beta + \beta^2 - 6 c \left(\alpha - c + \beta\right)\right\}$$

d1fdc = 2 * ρ * (c - α) * (β - c) * (α - 2 * c + β)
d2fdc = 2 * ρ * (α**2 + 4*α*β + β**2 - 6 * c * (α - c + β))

eom_c = TransientTerm(var=c) == DiffusionTerm(coeff=M, var=μ)

eom_μ =  ImplicitSourceTerm(coeff=1.0, var=μ) \
      == ImplicitSourceTerm(coeff=d2fdc, var=c) \
      +  d1fdc - d2fdc * c \
      -  DiffusionTerm(coeff=κ, var=c)

eom = eom_c & eom_μ

# Define initial conditions

def initialize(A, B):
    return ζ + ϵ * (
           cos(A[0] * x) * cos(B[0] * y) \
        + (cos(A[1] * x) * cos(B[1] * y))**2 \
        +  cos(A[2] * x  +     B[2] * y) \
        *  cos(A[3] * x  +     B[3] * y)
    )


if args.variant == "orig":
    # BM 1a specification: not periodic at all
    A0 = np.array([0.105, 0.130, 0.025, 0.070])
    B0 = np.array([0.110, 0.087,-0.150,-0.020])
elif args.variant == "peri":
    # Even integers as close to spec as achievable:
    # exactly periodic at the domain boundaries
    A0 = pi/Lx * np.array([6.0, 8.0, 2.0, 4.0])
    B0 = pi/Ly * np.array([8.0, 6.0,-10.,-2.0])
elif args.variant == "zany":
    # Perturbation of the periodic coefficients:
    # almost periodic, visually similar to the original
    A0 = pi/Lx * np.array([6.125, 7.875, 2.125, 4.125])
    B0 = pi/Ly * np.array([7.875, 5.125,-9.875,-1.875])
else:
    raise ValueError("Variant {} undefined.".format(args.variant))

# Initialize or reload field variables

if restartFromCheckpoint:
    mprint("Resuming simulation from {}".format(chk_file))

    load_restart(chk_file)

    # drop checkpoints we've already passed
    checkpoints = checkpoints[t < checkpoints]

    nrg_df = pd.read_csv(nrg_file)
    res_df = pd.read_csv(res_file)
else:
    c.value = initialize(A0, B0)
    μ.value = d1fdc[:]

    nrg_df = update_energy()
    res_df = None

c.updateOld()
μ.updateOld()

# === Create Viewers ===

fig, axs = plt.subplots(1, 2, figsize=(14, 6),
                        sharex=True, sharey=True,
                        constrained_layout=True)

fig.suptitle(r"$t = 0$")

viewers = MultiViewer([Viewer(vars=(c,), title=r"$c$",
                              axes=axs[0],
                              datamin=α,
                              datamax=β,
                              figaspect=1.0, legend=None),
                       Viewer(vars=(μ,), title=r"$\mu$",
                              axes=axs[1],
                              figaspect=1.0, legend=None,
                              cmap=plt.cm.viridis_r)])

# === Evolve the Equations of Motion ===

solver = Solver()

write_plot(fig, viewers)

def stepper_loop(check):
    global dt
    global t
    global nrg_df
    global res_df

    progress_bar = tqdm(PIDStepper(start=check.begin,
                                   stop=check.end,
                                   size=dt))

    n_sweep = 5

    res_t = [0.0] * n_sweep
    res_d = [0.0] * n_sweep
    res_s = [0.0] * n_sweep
    res_r = [0.0] * n_sweep

    for step in progress_bar:

        for i, sweep in enumerate(range(n_sweep)):
            label = "[{:12g}, {:12g}).{:02d}, Δt={:12g}".format(
                step.begin, check.end, sweep, step.size)
            progress_bar.set_description(label)
            res = eom.sweep(dt=step.size, solver=solver)
            res_t[i] = t
            res_d[i] = step.size
            res_s[i] = sweep
            res_r[i] = res

        victorious = step.succeeded(error=res/rtol)

        res_v = [victorious] * n_sweep

        res_row = pd.DataFrame(list(zip(res_t, res_d, res_s, res_r, res_v)),
                               columns=res_cols)
        res_df = pd.concat([res_df, res_row])

        if victorious:
            dt = step.size
            t += dt
            c.updateOld()
            μ.updateOld()
            nrg_df = update_energy(nrg_df)
        else:
            c.value = c.old
            μ.value = μ.old

        PETSc.garbage_cleanup()

    dt = step.want
    return parallel.bcast(nrg_df.energy_rate.iloc[-1])

def checkpoint_loop():
    for check in CheckpointStepper(start=t,
                                   stops=checkpoints,
                                   stop=t_fin):
        dFv_dt = stepper_loop(check)
        _ = check.succeeded()

        write_plot(fig, viewers)
        write_dataframe(nrg_file, nrg_df)
        write_dataframe(res_file, res_df)
        dump_restart(chk_file)

        gc.collect()

        # Endpoint detection: volume-weighted time rate of change of
        # the free energy, per <https://doi.org/10.1016/j.commatsci.2016.09.022>
        if t > t_min and dFv_dt < f_fin:
            mprint("Endpoint condition achieved: δFᵥ/δt = {} < {}".format(dFv_dt, f_fin))
            break


# Do it!
checkpoint_loop()

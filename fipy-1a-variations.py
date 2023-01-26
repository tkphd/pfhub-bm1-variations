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
import os
import pandas as pd
import psutil

from argparse import ArgumentParser

from fipy import CellVariable
from fipy import DiffusionTerm, ImplicitSourceTerm, TransientTerm
from fipy import numerix, parallel, Viewer
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

parser.add_argument("variant", help="one of 'orig', 'peri', or 'zany'")

args = parser.parse_args()

rank = parallel.procID
rankIsHead = (rank == 0)

if not rankIsHead:
    raise RuntimeError("ParallelGrid2D does not support parallel execution.")

iodir = args.variant
if not os.path.exists(iodir):
    os.mkdir(iodir)

ceil = numerix.ceil
cos  = numerix.cos
pi   = numerix.pi
log10= numerix.log10

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

### Prepare free energy output

columns = [
    "wall_time",
    "time",
    "free_energy",
    "mem_MB",
    "timestep",
    "mass",
    "energy_rate"
]

fcsv = "{}/energy.csv".format(iodir) if rankIsHead else None
fnpz = "{}/checkpoint.npz".format(iodir) if rankIsHead else None

restartFromCheckpoint = os.path.exists(fnpz)

# === Utility Functions ===

def dump_restart(fnpz=None):
    numerix.savez_compressed(fnpz, c=c.value, u=μ.value, t=t, dt=dt)

def load_restart(fnpz=None):
    global t
    # global dt
    global c
    global μ

    with numerix.load(fnpz) as data:
        c.value[:] = data["c"]
        μ.value[:] = data["u"]
        t = data["t"]
        # dt = data["dt"]

def fbulk(C):
    return ρ * (C - α)**2 * (β - C)**2

def energy_rate(df=None):
    if df is None or len(df) < 2:
        return 1.0
    V = Lx * Ly
    delF = float(df.free_energy.iloc[-1] - df.free_energy.iloc[-2])

    return -delF / (V * dt)

def mprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

def update_energy(df=None):
    firstRow = (df is None)
    # Integration of fields
    nrg = float((fbulk(c) + 0.5 * κ * (c.grad.mag)**2).sum())
    mas = c.sum()
    mem = parallel.allgather(proc.memory_info().rss) / 1024**2
    if rankIsHead:
        dFv_dt = float(energy_rate(df))
        timer = time.time() - startTime
        vals = (timer, t, nrg, mem, dt, mas, dFv_dt)
        index = [0] if firstRow else [len(df)]

        update = pd.DataFrame([vals], columns=columns, index=index)

        if firstRow:
            return update

        return pd.concat([df, update])

    return None

def write_energy(fcsv=None, df=None):
    if rankIsHead and df is not None and fcsv is not None:
        df.to_csv(fcsv, index=False)

def write_plot():
    imgname = "%s/spinodal.%08d.png" % (iodir, int(t))
    if rankIsHead and not os.path.exists(imgname):
        viewer.title = r"$t = %12g$" % t
        viewer.datamin = float(c.min())
        viewer.datamax = float(c.max())
        viewer.plot(filename=imgname)

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

t_fin = 20_000_000  # nothing should take this long
t_min = 100_000     # nothing should end before this
f_fin = 1e-16  # final rate of free energy evolution

# Write to disk uniformly in logarithmic space
checkpoints = numerix.unique(
    [
        int(float(10**q)) for q in
        numerix.arange(0, log10(t_fin), 0.1)
    ]
)

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
    A0 = numerix.array([0.105, 0.130, 0.025, 0.070])
    B0 = numerix.array([0.110, 0.087,-0.150,-0.020])
elif args.variant == "peri":
    # Even integers as close to spec as achievable:
    # exactly periodic at the domain boundaries
    A0 = pi/Lx * numerix.array([6.0, 8.0, 2.0, 4.0])
    B0 = pi/Ly * numerix.array([8.0, 6.0,-10.,-2.0])
elif args.variant == "zany":
    # Perturbation of the periodic coefficients:
    # almost periodic, visually similar to the original
    A0 = pi/Lx * numerix.array([6.125, 7.875, 2.125, 4.125])
    B0 = pi/Ly * numerix.array([7.875, 5.125,-9.875,-1.875])
else:
    raise ValueError("Variant {} undefined.".format(args.variant))

# Initialize or reload field variables

if restartFromCheckpoint:
    mprint("Resuming simulation from {}".format(fnpz))

    load_restart(fnpz)

    # drop checkpoints we've already passed
    checkpoints = checkpoints[t < checkpoints]

    nrg_df = pd.read_csv(fcsv)
else:
    c.value = initialize(A0, B0)
    μ.value = d1fdc[:]

    nrg_df = update_energy()

c.updateOld()
μ.updateOld()


# === Evolve the Equations of Motion ===

viewer = Viewer(vars=(c,))
solver = Solver()

write_plot()

def stepper(check):
    global dt
    global t
    global nrg_df

    progress_bar = tqdm(PIDStepper(start=check.begin,
                                   stop=check.end,
                                   size=dt,
                                   limiting=True))

    for step in progress_bar:
        res = 1.0
        swp = 0

        while swp < 10 and res > rtol:
            label = "Sweep {} [{:12g}, {:12g}), Δt={:12g}".format(
                swp, step.begin, check.end, step.size)
            progress_bar.set_description(label)
            res = eom.sweep(dt=step.size, solver=solver)
            swp += 1

        if step.succeeded(error=res/rtol):
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

def checkers():
    for check in CheckpointStepper(start=t,
                                   stops=checkpoints,
                                   stop=t_fin):
        dFv_dt = stepper(check)
        _ = check.succeeded()

        write_plot()
        write_energy(fcsv, nrg_df)
        dump_restart(fnpz)

        gc.collect()

        # Endpoint detection: volume-weighted time rate of change of
        # the free energy, per <https://doi.org/10.1016/j.commatsci.2016.09.022>
        if t > t_min and dFv_dt < f_fin:
            mprint("Endpoint condition achieved: δFᵥ/δt = {} < {}".format(dFv_dt, f_fin))
            break


# Do it!
checkers()

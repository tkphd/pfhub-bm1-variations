#!/usr/bin/python3
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

from fipy.solvers.petsc import LinearLUSolver as Solver
from fipy.solvers.petsc.comms import petscCommWrapper

from petsc4py import PETSc

try:
    from rich import print
except ImportError:
    pass

from steppyngstounes import CheckpointStepper, PIDStepper

from tqdm import tqdm


def mprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


parser = ArgumentParser(
    prog = 'fipy-bm1-variations',
    description = 'PFHub BM1 in FiPy with periodic initial condition variations'
)

parser.add_argument("variant", help="one of 'orig', 'peri', or 'zany'")
args = parser.parse_args()

iodir = args.variant

if not os.path.exists(iodir):
    os.mkdir(iodir)

mpl.use("agg")

ceil = numerix.ceil
cos  = numerix.cos
pi   = numerix.pi
log10= numerix.log10

proc = psutil.Process()

comm = petscCommWrapper.PETScCommWrapper()
rank = parallel.procID

### Prepare mesh & phase field

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

t_fin = 20_000_000
fin10 = log10(t_fin)

# Write to disk uniformly in logarithmic space
chkpts = [
    int(float(10**q)) for q in
    numerix.arange(0, fin10, 0.04)
]
chkpts = numerix.unique(chkpts)

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

fbulk = ρ * (c - α)**2 * (β - c)**2
d1fdc = 2 * ρ * (c - α) * (β - c) * (α - 2 * c + β)
d2fdc = 2 * ρ * (α**2 + 4*α*β + β**2 - 6 * c * (α - c + β))

eom_c = TransientTerm(var=c) == DiffusionTerm(coeff=M, var=μ)

eom_μ = ImplicitSourceTerm(coeff=1.0, var=μ) \
      == d1fdc - d2fdc * c \
      + ImplicitSourceTerm(coeff=d2fdc, var=c) \
      - DiffusionTerm(coeff=κ, var=c)

eom = eom_c & eom_μ


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

c.value = initialize(A0, B0)
μ.value = d1fdc[:]

c.updateOld()
μ.updateOld()


### Prepare free energy output

labs = [
    "wall_time",
    "time",
    "free_energy",
    "mem_GB",
    "timestep",
    "mass"
]

fcsv = "{}/energy.csv".format(iodir) if rank == 0 else None
nrg_df = None


def write_energy(fcsv=None, df=None):
    if rank == 0 and df is not None and fcsv is not None:
        df.to_csv(fcsv, index=False)


def update_energy(df=None):
    # Integration of fields: CellVolumeAverage, .sum(),
    nrg = (fbulk + 0.5 * κ * numerix.absolute(c.grad)**2).sum()
    mas = c.sum()
    mem = comm.allgather(proc.memory_info().rss) / 1024**3
    if rank == 0:
        timer = time.time() - startTime
        vals = (timer, t, nrg, mem, dt, mas)
        index = [0] if (df is None) else [len(df)]

        update = pd.DataFrame([vals], columns=labs, index=index)

        if df is None:
            return update

        return pd.concat([df, update])

    return None


def energy_rate(df=None):
    if df is not None:
        delF = (df["free_energy"].iloc[-1] - df["free_energy"].iloc[-2])
        delT = (df["time"].iloc[-1] - df["time"].iloc[-2])
        inV = 1 / (Lx * Ly)

        return inV * (delF / delT )

    return None


nrg_df = update_energy(nrg_df)

### Timestepping

rtol = 1e-4
solver = Solver()
viewer = Viewer(vars=(c,), title="$t = 0$")

def write_plot():
    imgname = "%s/spinodal.%08d.png" % (iodir, int(t))
    if rank == 0 and not os.path.exists(imgname):
        viewer.title = r"$t = %12g$" % t
        viewer.datamin = float(c.min())
        viewer.datamax = float(c.max())
        viewer.plot(filename=imgname)


write_plot()

def stepper(check):
    global dt
    global t
    global nrg_df

    label = "    Stepping [{:10g} .. {:10g}) / {:10g}".format(
        float(check.begin),
        float(check.end),
        float(check.size)
    )

    for step in tqdm(PIDStepper(start=check.begin,
                                stop=check.end,
                                size=dt),
                     desc=label):
        res = 1.0
        swp = 0

        while swp < 5 and res > rtol:
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


def checkers():
    for check in CheckpointStepper(start=0.0,
                                   stops=chkpts,
                                   stop=t_fin):
        stepper(check)
        _ = check.succeeded()

        write_plot()
        write_energy(fcsv, nrg_df)
        dF_dt = energy_rate(nrg_df)

        gc.collect()

        # Endpoint detection: volume-weighted time rate of change of
        # the free energy, per <https://doi.org/10.1016/j.commatsci.2016.09.022>
        if dF_dt is not None and dF_dt < 1e-14:
            break

# === Evolve the Equations of Motion ===

checkers()

# === Plot Variables ===
if rank == 0:
    from plot_energy import read_and_plot
    read_and_plot(iodir)

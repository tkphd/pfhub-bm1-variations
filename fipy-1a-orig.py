#!/Usr/bin/env python
# coding: utf-8

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
import psutil

from fipy import CellVariable
from fipy import DiffusionTerm, ImplicitSourceTerm, TransientTerm
from fipy import numerix, parallel, Viewer

from fipy import PeriodicGrid2D as Grid2D

from fipy.solvers.petsc import LinearLUSolver as Solver
from fipy.solvers.petsc.comms import petscCommWrapper

from math import log10

from petsc4py import PETSc

try:
    from rich import print
except ImportError:
    pass

from steppyngstounes import CheckpointStepper, PIDStepper

from argparse import ArgumentParser

parser = ArgumentParser(
    prog = 'fipy-bm1-variations',
    description = 'PFHub BM1 in FiPy with periodic initial condition variations'
)

parser.add_argument("variant", help="one of 'orig', 'peri', or 'zany'")
args = parser.parse_args()

def mprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


mpl.use("agg")

cos  = numerix.cos
pi   = numerix.pi

proc = psutil.Process()

comm = petscCommWrapper.PETScCommWrapper()
rank = parallel.procID
# cpus = parallel.Nproc

### Prepare mesh & phase field

Lx = Ly = 200

# The interface width 𝑙=4.47 units, so 10 points through the interface means dx~0.5?
dx = dy = 0.3125  # 640×640
# dx = dy = 0.2  # 1000×1000

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
dt = 1e-6

fin = 1500000
fin10 = log10(fin)

# Write to disk uniformly in logarithmic space
chkpts = [
    int(float(10**q)) for q in
    numerix.arange(0, fin10, 0.005)
]
if chkpts[-1] < fin:
    chkpts.append(fin)

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
     == (d1fdc - d2fdc * c) \
      + ImplicitSourceTerm(coeff=d2fdc, var=c) \
      - DiffusionTerm(coeff=κ, var=c)

eom = eom_c & eom_μ


### Initial Conditions -- As Specified

iodir = args.variant

if not os.path.exists(iodir):
    os.mkdir(iodir)


def initialize(A, B):
    return ζ + ϵ * (
           cos(A[0] * x) * cos(B[0] * y) \
        + (cos(A[1] * x) * cos(B[1] * y))**2 \
        +  cos(A[2] * x  +     B[2] * y) \
        *  cos(A[3] * x  +     B[3] * y)
    )


if args.variant == "orig":
    # BM 1a specification: not periodic at all
    c.value = initialize(
        (0.105, 0.130, 0.025, 0.070),
        (0.110, 0.087,-0.150,-0.020)
    )
else:
    raise ValueError("No such condition: {}".format(args["variant"]))


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

if rank == 0:  # write the CSV header
    fcsv = "{}/energy.csv".format(iodir)
    with open(fcsv, "w") as fh:
        fh.write("{},{},{},{},{},{}\n".format(*labs))
else:
    fcsv = None


def update_energy(fh=None):
    # Integration of fields: CellVolumeAverage, .sum(),
    nrg = (fbulk - 0.5 * κ * numerix.dot(c.grad, c.grad)).sum()
    mas = c.sum()
    mem = comm.allgather(proc.memory_info().rss) / 1024**3
    if rank == 0:
        timer = time.time() - startTime

        vals = [timer, t, nrg, mem, dt, mas]

        with open(fcsv, "a") as fh:
            fh.write("{},{},{},{},{},{}\n".format(*vals))


update_energy(fcsv)


### Timestepping

rtol = 1e-3
solver = Solver()
viewer = Viewer(vars=(c,), title="$t = 0$")

def write_plot():
    imgname = "%s/spinodal.%08d.png" % (iodir, int(t))
    if rank == 0 and not os.path.exists(imgname):
        viewer.title = r"$t = %12g$" % t
        viewer.datamin = float(c.min())
        viewer.datamax = float(c.max())
        # cb_ticks = viewer.colorbar.get_ticks().tolist()
        # cb_ticks[0] = round(float(c.min()), 4)
        # cb_ticks.append(round(float(c.max()), 4))
        # viewer.colorbar.set_ticks(cb_ticks)
        viewer.plot(filename=imgname)


# mprint("Writing a checkpoint at the following times:")
# mprint(chkpts)

write_plot()

def stepper(check):
    global dt
    global t

    for step in PIDStepper(start=check.begin,
                           stop=check.end,
                           size=dt):
        mprint("    Stepping [{:12g} .. {:12g}) / {:12g}".format(float(step.begin),
                                                                 float(step.end),
                                                                 float(step.size)),
               end=" ")

        for sweep in range(2):
            res = eom.sweep(dt=step.size, solver=solver)

        if step.succeeded(error=res/rtol):
            mprint("✔")
            dt = step.size
            t += dt
            c.updateOld()
            μ.updateOld()
            update_energy(fcsv)
        else:
            mprint("✘")
            c.value = c.old
            μ.value = μ.old

        PETSc.garbage_cleanup()

    dt = step.want

    write_plot()


def checkers():
    global dt
    global t
    for check in CheckpointStepper(start=0.0,
                                   stops=chkpts,
                                   stop=fin):
        mprint("Launching [{:12g} .. {:12g})".format(check.begin,
                                                     check.end))

        stepper(check)

        _ = check.succeeded()

        gc.collect()


checkers()
plt.close()

# === Plot Variables ===
if rank == 0:
    from plot_energy import read_and_plot
    read_and_plot(iodir)

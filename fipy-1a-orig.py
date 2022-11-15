#!/usr/bin/env python
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

from memory_profiler import profile

import gc
import os
import psutil
import time

from fipy import PeriodicGrid2D as Grid2D

from fipy import CellVariable
from fipy import DiffusionTerm, ImplicitSourceTerm, TransientTerm
from fipy import numerix, parallel

from fipy.solvers.petsc import LinearLUSolver as Solver
from fipy.solvers.petsc.comms import petscCommWrapper

from math import ceil, log10

from steppyngstounes import CheckpointStepper, PIDStepper

try:
    from rich import print
except ImportError:
    pass

try:
    startTime = time.time_ns()
    time_has_ns = True
except AttributeError:
    startTime = time.time()
    time_has_ns = False

cos = numerix.cos
pi = numerix.pi
proc = psutil.Process()

comm = petscCommWrapper.PETScCommWrapper()
rank = parallel.procID

def mprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

### Prepare mesh & phase field

Lx = Ly = 200
dx = dy = 0.3125

mesh = Grid2D(nx=Lx, ny=Ly, dx=dx, dy=dy)
x, y = mesh.cellCenters

c = CellVariable(mesh=mesh, name=r"$c$",   hasOld=True)
μ = CellVariable(mesh=mesh, name=r"$\mu$", hasOld=True)

#### Set thermo-kinetic constants from the BM1 specification

α = 0.3
β = 0.7
ρ = 5
κ = 2
M = 5

t = 0.0
dt = 1e-5
fin = 0.05

# Write to disk every 1, 2, 5, 10, 20, 50, ...
chkpts = [float(p * 10**q) \
          for q in range(-2, ceil(log10(fin + 1.0e-6))) \
          for p in (1, 2, 5)]

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

"""
## Uncomment and run this cell to double-check the derivatives.

import sympy.abc
from sympy import Eq, diff, expand, factor, symbols
fbulk = sympy.abc.rho * (sympy.abc.c - sympy.abc.alpha)**2 \
                      * (sympy.abc.beta - sympy.abc.c)**2

display(Eq(symbols("f"), fchem))
display(Eq(symbols("f'"), factor(diff(fchem,
                                      sympy.abc.c))))
display(Eq(symbols("f''"), factor(expand(diff(fbulk,
                                             sympy.abc.c,
                                             sympy.abc.c)))))
"""

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

if rank == 0:
    iodir = "orig"

    if not os.path.exists(iodir):
        os.mkdir(iodir)

c0 = 0.5
ϵ  = 0.01

def initialize(A, B):
    return c0 + ϵ * (
           cos(A[0] * x) * cos(B[0] * y) \
        + (cos(A[1] * x) * cos(B[1] * y))**2 \
        +  cos(A[2] * x  +     B[2] * y) \
        *  cos(A[3] * x  +     B[3] * y)
    )


# BM 1a specification: not periodic at all

A0 = [0.105, 0.130, 0.025, 0.070]
B0 = [0.110, 0.087,-0.150,-0.020]

c.value = initialize(A0, B0)

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
        if time_has_ns:
            timer = 1e-9 * (time.time_ns() - startTime)
        else:
            timer = time.time() - startTime

        vals = [timer, t, nrg, mem, dt, mas]

        with open(fcsv, "a") as fh:
            fh.write("{},{},{},{},{},{}\n".format(*vals))

update_energy(fcsv)


### Timestepping

rtol = 1e-3
solver = Solver()

mprint("Writing a checkpoint at the following times:")
mprint(chkpts)

@profile
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

        gc.collect()

    dt = step.want

@profile
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

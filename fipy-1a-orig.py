#!/usr/bin/env python
# coding: utf-8

# # PFHub BM 1a in FiPy with Steppyngstounes
# 
# This notebook implements [PFHub] Benchmark [1a][spinodal] using [FiPy] and [steppyngstounes].
# It also explores alternative initial conditions that are more-periodic than the specification.
# 
# <!-- links -->
# [FiPy]: https://www.ctcms.nist.gov/fipy
# [spinodal]: https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/#(a)-Square-periodic
# [steppyngstounes]: https://github.com/usnistgov/steppyngstounes
# [PFHub]: https://pages.nist.gov/pfhub/

# In[ ]:


import gc
import os
import psutil
import time

from fipy import Variable, FaceVariable, CellVariable
from fipy import DiffusionTerm, ImplicitSourceTerm, TransientTerm
from fipy import numerix, parallel, Viewer

from fipy import PeriodicGrid2D as Grid2D
from fipy.solvers.petsc import LinearLUSolver as Solver
from fipy.solvers.petsc.comms import petscCommWrapper

from steppyngstounes import CheckpointStepper, PIDStepper


# In[ ]:


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
procID = parallel.procID


# ## Prepare mesh & phase field

# In[ ]:


nx = ny = 200
dx = dy = 1
Lx = nx * dx
Ly = ny * dy

mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
x, y = mesh.cellCenters

c = CellVariable(mesh=mesh, name=r"$c$",   value=0.0, hasOld=True)
μ = CellVariable(mesh=mesh, name=r"$\mu$", value=0.5, hasOld=True)


# ### Set thermo-kinetic constants from the BM1 specification

# In[ ]:


α = 0.3
β = 0.7
ρ = 5
κ = 2
M = 5


# ## Define equations of motion
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

# In[ ]:


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

# In[ ]:


fbulk = ρ * (c - α)**2 * (β - c)**2
d1fdc = 2 * ρ * (c - α) * (β - c) * (α - 2 * c + β)
d2fdc = 2 * ρ * (α**2 + 4*α*β + β**2 - 6 * c * (α - c + β))


# In[ ]:


eom_c = TransientTerm(var=c) == DiffusionTerm(coeff=M, var=μ)

eom_μ = ImplicitSourceTerm(coeff=1.0, var=μ) \
     == (d1fdc - d2fdc * c) \
      + ImplicitSourceTerm(coeff=d2fdc, var=c) \
      - DiffusionTerm(coeff=κ, var=c)

eom = eom_c & eom_μ


# ## Initial Conditions -- As Specified

# In[ ]:


iodir = "1a.0"

if not os.path.exists(iodir):
    os.mkdir(iodir)


# In[ ]:


t = Variable(0.0)
dt = Variable(1e-5)

c0 = 0.5
ϵ = 0.01

def initialize(A, B):
    return c0 + ϵ * (
           cos(A[0] * x) * cos(B[0] * y) \
        + (cos(A[1] * x) * cos(B[1] * y))**2 \
        +  cos(A[2] * x  +     B[2] * y) \
        *  cos(A[3] * x  +     B[3] * y)
    )


# In[ ]:


# BM 1a specification: not periodic at all

A0 = [0.105, 0.130, 0.025, 0.070]
B0 = [0.110, 0.087,-0.150,-0.020]

c.value = initialize(A0, B0)

c.updateOld()
μ.updateOld()


# ## Prepare free energy output

# In[ ]:


df = None

labs = [
    "wall_time", 
    "time",     
    "free_energy",
    "mem_GB",    
    "timestep", 
    "mass"
]

if procID == 0:  # write the CSV header
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
    if procID == 0:
        if time_has_ns:
            timer = 1e-9 * (time.time_ns() - startTime)
        else:
            timer = time.time() - startTime
    
        vals = [timer, t, nrg, mem, dt, mas]
        
        with open(fcsv, "a") as fh:
            fh.write("{},{},{},{},{},{}\n".format(*vals))
        
update_energy(fcsv)


# ## Timestepping

# In[ ]:


fin = 2e6
rtol = 1e-3
solver = Solver()

# Write to disk every 1, 2, 5, 10, 20, 50, ...
chkpts = [p * 10**q \
          for q in range(3) \
          for p in (1, 2, 5)]

print("Writing a checkpoint at the following times:")
print(chkpts)


# In[ ]:


for check in CheckpointStepper(start=0.0,
                               stops=chkpts,
                               stop=fin):
    for step in PIDStepper(start=check.begin,
                           stop=check.end, 
                           size=dt):
        for sweep in range(2):
            res = eom.sweep(dt=step.size, solver=solver)
    
        if step.succeeded(error=res/rtol):
            t.value = step.end
            dt = step.size
            c.updateOld()
            μ.updateOld()
            # df = update_energy()
            update_energy(fcsv)
        else:
            c.value = c.old
            μ.value = μ.old

        gc.collect()
        
    if check.succeeded():
        dt = step.want
        # df.to_csv("{}/energy.csv.gz".format(iodir))
        # write_img()
        # update_display(df.tail(), display_id="nrg")
        # update_display(viewer.plot(), display_id="plt")
        # viewer.title = r"$t=%g$" % t
        # viewer.plot()


# In[ ]:





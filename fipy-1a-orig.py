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


get_ipython().run_line_magic('matplotlib', 'inline')
import gc
import ipywidgets as widgets
import matplotlib.pyplot as plt
import os
import pandas as pd
import psutil
import sympy.abc

from IPython.display import display, update_display
from sympy import Eq, diff, expand, factor, symbols
from tqdm import tqdm
import time

from steppyngstounes import CheckpointStepper, PIDStepper

from fipy import Variable, FaceVariable, CellVariable
from fipy import DiffusionTerm, ImplicitSourceTerm, TransientTerm
from fipy import numerix, parallel, Viewer

from fipy import PeriodicGrid2D as Grid2D
from fipy.solvers.petsc import LinearLUSolver as Solver

cos = numerix.cos
pi = numerix.pi
proc = psutil.Process()


# In[ ]:


try:
    startTime = time.time_ns()
    time_has_ns = True
except AttributeError:
    startTime = time.time()
    time_has_ns = False


# ## Prepare mesh & phase field

# In[ ]:


nx = ny = 200
dx = dy = 1
Lx = nx * dx
Ly = ny * dy
mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)


# In[ ]:


x, y = mesh.cellCenters
c = CellVariable(mesh=mesh, name=r"$c$",   value=0.0, hasOld=True)
μ = CellVariable(mesh=mesh, name=r"$\mu$", value=0.5, hasOld=True)


# ## Define equations of motion
# 
# This is based on [fipy.examples.cahnHilliard.mesh2DCoupled](https://www.ctcms.nist.gov/fipy/examples/cahnHilliard/generated/examples.cahnHilliard.mesh2DCoupled.html). The coupled equations are
# 
# $$
# \frac{\partial c}{\partial t} = \nabla \cdot \nabla\mu
# $$
# 
# $$
# \mu = \left(\frac{\partial f}{\partial c} - \frac{\partial^2 f}{\partial c^2} \cdot c\right)_{\mathrm{old}} + \frac{\partial^2 f}{\partial c^2}\cdot c - \kappa \nabla^2 c
# $$
# 
# where the second term on $\mu$ is an `ImplicitSourceTerm` and the last is a `DiffusionTerm`.

# In[ ]:


fchem = sympy.abc.rho * (sympy.abc.c - sympy.abc.alpha)**2 \
                      * (sympy.abc.beta - sympy.abc.c)**2

display(Eq(symbols("f"), fchem))
display(Eq(symbols("f'"), factor(diff(fchem,
                                     sympy.abc.c))))
display(Eq(symbols("f''"), factor(expand(diff(fchem,
                                             sympy.abc.c,
                                             sympy.abc.c)))))


# $$f_{\mathrm{c}} = \rho (c - \alpha)^2 (\beta - c)^2$$
# 
# $$f'_{\mathrm{c}} = 2\rho (c - \alpha)(\beta - c)(\alpha - 2c + \beta)$$
# 
# $$f''_{\mathrm{c}} = 2\rho\left\{\alpha^2 + 4 \alpha \beta + \beta^2 - 6 c \left(\alpha - c + \beta\right)\right\}$$

# In[ ]:


α = 0.3
β = 0.7
ρ = 5

fc = ρ * (c - α)**2 * (β - c)**2
dfdc = 2 * ρ * (c - α) * (β - c) * (α - 2 * c + β)
d2fc = 2 * ρ * (α**2 + 4*α*β + β**2 \
                  - 6 * c * (α - c + β))


# In[ ]:


κ = 2
M = 5

eom_c = TransientTerm(var=c) == DiffusionTerm(coeff=M, var=μ)
eom_μ = ImplicitSourceTerm(coeff=1.0, var=μ) \
     == (dfdc - d2fc * c) \
      + ImplicitSourceTerm(coeff=d2fc, var=c) \
      - DiffusionTerm(coeff=κ, var=c)

eom = eom_c & eom_μ


# ## Initial Conditions

# In[ ]:


t = Variable(0)
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

iodir = "1a.0"

if not os.path.exists(iodir):
    os.mkdir(iodir)


# In[ ]:


viewer = Viewer(vars=(c,), datamin=0., datamax=1., cmap="coolwarm")

def write_img():
    viewer.title = r"$t=%g$" % t
    viewer.plot("{iodir}/comp_{t:0>12.0f}.png")
    
write_img()


# ## Prepare free energy output

# In[ ]:


df = None

def update_energy():
    nrg = numerix.sum(fc - 0.5 * κ * numerix.dot(c.grad, c.grad))
    mas = numerix.sum(c)
    mem = proc.memory_info().rss / 1024**2
    if time_has_ns:
        timer = 1e-9 * (time.time_ns() - startTime)
    else:
        timer = time.time() - startTime
    if parallel.procID == 0:
        labs = ["wall_time", "time", "free_energy",
                "mem_MB", "timestep", "mass"]
        vals = [[timer, t, nrg, mem, dt, mas]]
        indx = [0] if df is None \
                   else [len(df)]
        
        return pd.concat([df,
                          pd.DataFrame(vals,
                                       columns=labs,
                                       index=indx)]
                        )
   
df = update_energy()

# disp_nrg = display(df.tail(), display_id="nrg")


# ## Timestepping

# In[ ]:


fin = 2e6
rtol = 1e-3
solver = Solver()

# Write an image every 1, 2, 5, 10, 20, 50, ...
chkpts = [p * 10**q \
          for q in range(6) \
          for p in (1, 2, 5)]


# In[ ]:


for check in CheckpointStepper(start=0.0,
                               stops=chkpts
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
            df = update_energy()
        else:
            c.value = c.old
            μ.value = μ.old

        gc.collect()
        
    if check.succeeded():
        dt = step.want
        df.to_csv("{iodir}/energy.csv.gz")
        write_img()
        # update_display(df.tail(), display_id="nrg")
        # update_display(viewer.plot(), display_id="plt")
        # viewer.title = r"$t=%g$" % t
        # viewer.plot()


# In[ ]:





#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from math import pi

from rich import print

from sympy import init_printing, pretty
from sympy import factor, lambdify, symbols
from sympy import Eq, N, cos
from sympy.abc import x, y

init_printing()

# === create the spatial grid ===

Lx = 200
Ly = 200

X = np.linspace(0, Lx, 2 * Lx + 1)
Y = np.linspace(0, Ly, 2 * Ly + 1)
X1, Y1 = np.meshgrid(X, Y)

# === create the equations ===

a1, a2, a3, a4 = symbols("a1 a2 a3 a4")
b1, b2, b3, b4 = symbols("b1 b2 b3 b4")

c1 =  cos(a1*x) * cos(b1*y)
c2 = (cos(a2*x) * cos(b2*y))**2
c3 =  cos(a3*x  +     b3*y) * \
      cos(a4*x  +     b4*y)

C0 = factor(0.5 + 0.01 * (c1 + c2 + c3))

print(pretty(Eq(symbols("C"), C0), use_unicode=True))

# === substitute original values ===

Q0 = factor(C0.subs({a1: 0.105,
                     a2: 0.130,
                     a3: 0.025,
                     a4: 0.070,
                     b1: 0.110,
                     b2: 0.087,
                     b3:-0.150,
                     b4:-0.020})
            )

F0 = lambdify((x, y), Q0)
print(pretty(Eq(symbols("C0"), Q0), use_unicode=True))

# === substitute periodic values ===

# Exact coefficients:
# A = [6.68, 8.28, 1.59, 4.46]
# B = [7.00, 5.54,-9.55,-1.27]

A = [6, 8,  2, 4]
B = [8, 6,-10,-2]

Q1 = factor(C0.subs({a1: N(A[0] * pi/Lx, 3),
                     a2: N(A[1] * pi/Lx, 3),
                     a3: N(A[2] * pi/Lx, 3),
                     a4: N(A[3] * pi/Lx, 3),
                     b1: N(B[0] * pi/Ly, 3),
                     b2: N(B[1] * pi/Ly, 3),
                     b3: N(B[2] * pi/Ly, 3),
                     b4: N(B[3] * pi/Ly, 3)})
            )

print(pretty(Eq(symbols("C1"), Q1), use_unicode=True))

F1 = lambdify((x, y),
              C0.subs({a1: A[0] * pi/Lx,
                       a2: A[1] * pi/Lx,
                       a3: A[2] * pi/Lx,
                       a4: A[3] * pi/Lx,
                       b1: B[0] * pi/Ly,
                       b2: B[1] * pi/Ly,
                       b3: B[2] * pi/Ly,
                       b4: B[3] * pi/Ly})
              )

# === substitute perturbed values ===

C = [6.125, 7.875, 2.125, 4.125]
D = [7.875, 5.125,-9.875,-1.875]

Q2 = factor(C0.subs({a1: N(C[0] * pi/Lx, 3),
                     a2: N(C[1] * pi/Lx, 3),
                     a3: N(C[2] * pi/Lx, 3),
                     a4: N(C[3] * pi/Lx, 3),
                     b1: N(D[0] * pi/Ly, 3),
                     b2: N(D[1] * pi/Ly, 3),
                     b3: N(D[2] * pi/Ly, 3),
                     b4: N(D[3] * pi/Ly, 3)})
            )

print(pretty(Eq(symbols("C2"), Q2), use_unicode=True))

F2 = lambdify((x, y),
              C0.subs({a1: C[0] * pi/Lx,
                       a2: C[1] * pi/Lx,
                       a3: C[2] * pi/Lx,
                       a4: C[3] * pi/Lx,
                       b1: D[0] * pi/Ly,
                       b2: D[1] * pi/Ly,
                       b3: D[2] * pi/Ly,
                       b4: D[3] * pi/Ly})
              )

# === plot the thing ===

plt.figure()

# plt.plot(X, F0(X, 3), label="cx")
# plt.plot(X, F0(3, X), label="cy")

plt.plot(X, F1(X, 3), label="cx'")
plt.plot(X, F1(3, X), label="cy'")

plt.plot(X, F2(X, 3), label="cx''")
plt.plot(X, F2(3, X), label="cy''")

plt.legend(loc="best")
plt.savefig("ic-lines.png", dpi=400, bbox_inches="tight")
plt.close()

# === plot the whole field ===

Z0 = F0(X1, Y1)
Z1 = F1(X1, Y1)
Z2 = F2(X1, Y1)

ZE = (Z1 - Z0)**2
PE = (Z2 - Z0)**2

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15,10), sharex=True, sharey=True)
font = {"fontsize": 16}

ax = axs[0][0]
ax.set_title("original", fontdict=font)
ax.set_aspect("equal", adjustable='box')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ori = ax.imshow(Z0, vmin=Z0.min(), vmax=Z0.max(), cmap="coolwarm")
fig.colorbar(ori, ax=ax, shrink=0.4)

ax = axs[0][1]
ax.set_title("periodic", fontdict=font)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
per = ax.imshow(Z1, vmin=Z1.min(), vmax=Z1.max(), cmap="coolwarm")
fig.colorbar(per, ax=ax, shrink=0.4)

ax = axs[0][2]
ax.set_title("squared error", fontdict=font)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
err = ax.imshow(ZE, vmin=ZE.min(), vmax=ZE.max(), cmap="viridis")
fig.colorbar(err, ax=ax, shrink=0.4)

ax = axs[1][0]
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

ax = axs[1][1]
ax.set_title("perturbed", fontdict=font)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
per = ax.imshow(Z2, vmin=Z2.min(), vmax=Z2.max(), cmap="coolwarm")
fig.colorbar(per, ax=ax, shrink=0.4)

ax = axs[1][2]
ax.set_title("perturbed error", fontdict=font)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
err = ax.imshow(ZE, vmin=PE.min(), vmax=PE.max(), cmap="viridis")
fig.colorbar(err, ax=ax, shrink=0.4)

# === display & save ===

plt.tight_layout()
plt.savefig("ic.png", bbox_inches="tight", dpi=400)

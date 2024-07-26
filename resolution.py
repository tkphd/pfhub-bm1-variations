#!/usr/bin/python3
# Get mesh resolutions
import numpy as np
import pandas as pd
from rich import print
import os
import sys

sys.path.append(os.path.dirname(__file__))
from spectral import L, M, κ

def stable_k(h):
    # Return dt satisfying the CFL given dx
    # C = κ * M * dt / dx**(-4)
    k = h**4 / (M * κ)
    return 2**(np.floor(np.log2(k)))

hg = 0.0625

x = np.arange(hg, 50, hg)
y = np.array([ n for n in x if (L % n == 0)])

data = pd.DataFrame({"h": y})
data["k"] = data["h"].apply(stable_k)

print(data)

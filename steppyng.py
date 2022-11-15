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

from memory_profiler import profile

import psutil

from steppyngstounes import CheckpointStepper, PIDStepper

proc = psutil.Process()

# ## Timestepping

fin = 1
rtol = 1e-3

t = 0.0
dt = 1e-5

# Write to disk every 1, 2, 5, 10, 20, 50, ...
chkpts = [p * 10**q \
          for q in (-1, 0) \
          for p in (1, 2, 5)]

print("Writing a checkpoint at the following times:")
print(chkpts)

@profile
def run():
    global dt
    global t
    for check in CheckpointStepper(start=0.0,
                                   stops=chkpts,
                                   stop=1):
        print("Launching [{} .. {})".format(check.begin, check.end))

        for step in PIDStepper(start=check.begin,
                               stop=check.end,
                               size=dt):
            print("    Stepping [{} .. {})".format(step.begin, step.end))

            if step.succeeded(error=dt * 0.9):
                dt = step.size
                t  = step.end

        if check.succeeded():
            dt = step.want

run()

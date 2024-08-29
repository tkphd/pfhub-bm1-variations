#!/usr/bin/env python3

"""
Perform spectral interpolation of grid data by remapping the domain
from [0, L] to [0, 2π] and zero-padding in reciprocal space.
(Do not FFT-shift.)
Use the FFT to generate the power spectrum, $\tilde{y}^*\tilde{y}$.
Transform the power spectrum back to real space, producing the
autocorrelation (aka 2-pt stats).
"""

from argparse import ArgumentParser
import gc
import glob
import numpy as np
import os
from parse import compile
import sys
from tqdm.contrib.concurrent import process_map
import zipfile

# import from `spectral.py` in same folder as the script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from spectral.viz import FourierInterpolant, radial_profile

# set the ultimately refined mesh resolution
mesh_h = 2**(-4)            # 0.0625 = 2⁻⁴
mesh_N = int(200 / mesh_h)
if mesh_N % 2 != 0:
    raise ValueError("Reference mesh size is not even!")

parse_dt  = compile("dt{dt:8f}{suffix}")
parse_dx  = compile("{prefix}x{dx:8f}")
parse_dtx = compile("dt{dt:8f}_dx{dx:8f}")
parse_npz = compile("{prefix}/c_{t:8d}.npz")

def correlate(data):
    """Compute the auto-correlation / 2-point statistics of a field variable"""
    signal = data - data.mean()
    fft = np.fft.rfftn(signal)
    psd = fft * np.conjugate(fft)
    return np.fft.irfftn(psd).real / (np.var(signal) * signal.size)


def sim_details(iodir):
    dx = parse_dx.parse(iodir)["dx"]
    Nx = np.rint(200. / dx).astype(int)

    files = sorted(glob.glob(f"{iodir}/c_*.npz"))

    return {
        "dx": float(dx),
        "Nx": int(Nx),
        "files": files
    }


def upsampled(c_npz):
    jobdir = os.path.dirname(c_npz)
    c_name = os.path.basename(c_npz)
    t = parse_npz.parse(c_npz)["t"]
    k_npz = f"{jobdir}/interp/{c_name}"

    hi_res = None
    hi_fft = None
    gold_h = 0.0625
    mesh_N = int(3200 * gold_h / mesh_h)
    interpolant = FourierInterpolant((mesh_N, mesh_N))

    if not os.path.exists(k_npz):
        # Upsample and save spectrally interpolated mesh data
        try:
            with np.load(c_npz) as npz:
                lo_res = npz["c"]

            hi_res = interpolant.upsample(lo_res)
            signal = hi_res - hi_res.mean()
            hi_fft = np.fft.rfftn(signal)
            hi_psd = hi_fft * np.conjugate(hi_fft)
            hi_cor = np.fft.irfftn(hi_psd).real / (np.var(signal) * signal.size)
            cor_r, cor_μ = radial_profile(hi_cor)
            cor_r = gold_h * np.array(cor_r)

            np.savez_compressed(k_npz,
                                t=t,
                                dx=mesh_h,
                                c=hi_res,
                                k=hi_fft,
                                p=hi_psd,
                                a=hi_cor,
                                r=cor_r,
                                μ=cor_μ)
        except FileNotFoundError or zipfile.BadZipFile:
            pass


variant = os.path.basename(os.getcwd())

# parse command-line flags
parser = ArgumentParser()
parser.add_argument("--dx",
                    type=float,
                    default=2**(-4),
                    help="Reference mesh resolution (smaller than gold standard)")

# === Interpolate! ===

if __name__ == "__main__":
    args = parser.parse_args()

    dirs = sorted(glob.glob("dt?.??????_dx???.????"))
    jobs = {}

    for job in dirs:
        stats = sim_details(job)
        if stats["dx"] > mesh_h:
            jobs[job] = stats

    for jobdir, job_par in jobs.items():
        terpdir = f"{jobdir}/interp"
        if not os.path.exists(terpdir):
            os.mkdir(terpdir)

        process_map(upsampled,
                    job_par["files"],
                    max_workers=16,
                    **{"desc": jobdir})

        gc.collect()

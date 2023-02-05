# Makefile for PFHub BM 1 variations
# with periodic grids and serial solvers

TIMEFMT = '\n   %E〔%e𝑠 wall,  %U𝑠 user,  %S𝑠 sys,  %M KB,  %F faults,  %c switches〕'

.PHONY: clean orig peri zany viz mks-orig mks-peri mks-zany

# === FiPy ===

orig: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< --prefix fipy --variant $@

peri: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< --prefix fipy --variant $@

zany: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< --prefix fipy --variant $@

viz:
	/usr/bin/time -f $(TIMEFMT) ./plot_energy.py --directory fipy --platform FiPy

# === PyMKS ===

mks-orig: spectral-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< orig

mks-peri: spectral-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< peri

mks-zany: spectral-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< zany

# === Utilities ===

clean:
	rm -r orig/* peri/* zany/*

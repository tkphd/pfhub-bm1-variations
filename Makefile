# Makefile for PFHub BM 1 variations
# with periodic grids and serial solvers

TIMEIT = /usr/bin/time -f '\n   %E〔%e𝑠 wall,  %U𝑠 user,  %S𝑠 sys,  %M KB,  %F faults,  %c switches〕'

.PHONY: clean orig peri zany viz mks-clean mks-orig mks-peri mks-zany mks-viz

# === FiPy ===

orig: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< --prefix fipy --variant $@

peri: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< --prefix fipy --variant $@

zany: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< --prefix fipy --variant $@

viz:
	$(TIMEIT) ./plot_energy.py --directory fipy --platform FiPy

clean:
	rm -r fipy/orig/* fipy/peri/* fipy/zany/*

# === PyMKS ===

mks-orig: pymks-1a-variations.py
	OMP_NUM_THREADS=1 $(TIMEIT) python3 $< orig

mks-peri: pymks-1a-variations.py
	OMP_NUM_THREADS=1 $(TIMEIT) python3 $< peri

mks-zany: pymks-1a-variations.py
	OMP_NUM_THREADS=1 $(TIMEIT) python3 $< zany

mks-viz:
	$(TIMEIT) ./plot_energy.py --directory pymks --platform PyMKS

mks-clean:
	rm -r pymks/orig/* pymks/peri/* pymks/zany/*

# Makefile for PFHub BM 1 variations
# with periodic grids and serial solvers

.PHONY: clean orig peri zany viz mks-orig mks-peri mks-zany
TIMEFMT = '\n   %E 〔%e𝑠 wall,  %U𝑠 user,  %S𝑠 sys,  %M KB,  %F faults,  %c switches〕'

orig: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< $@

peri: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< $@

zany: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< $@

viz:
	/usr/bin/time -f $(TIMEFMT) ./plot_energy.py --directory fipy --platform FiPy --sweeps 5


mks-orig: spectral-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< orig

mks-peri: spectral-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< peri

mks-zany: spectral-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< zany

clean:
	rm -r orig/* peri/* zany/*

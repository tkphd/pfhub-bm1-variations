# Makefile for PFHub BM 1 variations
# with periodic grids and serial solvers

.PHONY: clean mon orig peri zany

orig: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< $@

peri: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< $@

zany: fipy-1a-variations.py
	OMP_NUM_THREADS=1 python3 $< $@

mon:
	watch -n 10 "xsv table orig/energy.csv | head -n 15; echo '...'; xsv table orig/energy.csv | tail -n 15"

clean:
	rm -r orig/* peri/* zany/*

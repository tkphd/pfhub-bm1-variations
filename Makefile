# Makefile for PFHub BM 1 variations

all: orig/energy.csv
.PHONY: all clean

%/energy.csv: fipy-1a-%.py
	mpirun -np 1 python3 -u $<

clean:
	rm -r orig

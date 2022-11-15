# Makefile for PFHub BM 1 variations

all: orig
.PHONY: all clean orig test watch

orig: fipy-1a-orig.py
	mpirun -np 1 python3 -u $< 2>orig/profile.log

test: steppyng.py
	python3 -u $<

mon:
	watch 'xsv table orig/energy.csv | head -n 20; echo; echo "..."; echo; xsv table orig/energy.csv | tail -n 20'

clean:
	rm -r orig

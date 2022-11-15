# Makefile for PFHub BM 1 variations

all: orig
.PHONY: all clean orig test watch

orig: fipy-1a-orig.py
	mkdir $@
	mpirun -np 1 python3 -u $< | tee $@/profile.log

mon:
	watch 'xsv table orig/energy.csv | head -n 20; echo; echo "..."; echo; xsv table orig/energy.csv | tail -n 20'

test: steppyng.py
	python3 -u $<

clean:
	rm -r orig

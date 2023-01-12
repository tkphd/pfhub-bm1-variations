# Makefile for PFHub BM 1 variations

FIPYLOG = $(HOME)/repositories/fipy/fipy/tools/logging
NPROC = 4

all: orig
.PHONY: all clean orig peri zany prof test watch

prof: fipy-1a-orig.py
	$$(mkdir -p $@)
	OMP_NUM_THREADS=1 \
	FIPY_LOG_CONFIG=$(FIPYLOG)/scattered_config.json \
	python3 -u $< orig | tee $@/profile.log
# mpirun -np $(NPROC)

orig: fipy-1a-orig.py
	$$(mkdir -p $@; rm -vf $@/*)
	OMP_NUM_THREADS=1 \
	python3 -u $< $@ | tee $@/profile.log

peri: fipy-1a-orig.py
	$$(mkdir -p $@; rm -vf $@/*)
	OMP_NUM_THREADS=1 \
	python3 -u $< $@ | tee $@/profile.log

zany: fipy-1a-orig.py
	$$(mkdir -p $@; rm -vf $@/*)
	OMP_NUM_THREADS=1 \
	python3 -u $< $@ | tee $@/profile.log

mon:
	watch 'xsv table orig/energy.csv | head -n 20; echo; echo "..."; echo; xsv table orig/energy.csv | tail -n 20'

test: fipy-1a-orig.py
	$$(mkdir -p $@)
	OMP_NUM_THREADS=1 \
	FIPY_LOG_CONFIG=$(FIPYLOG)/scattered_config.json \
	mprof run $< orig | tee $@/profile.log

clean:
	rm -r orig

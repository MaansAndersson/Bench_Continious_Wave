# 3D PEC cavity benchmark: solve_cw vs solve_wh (MeepHoltz fork), C++ and Python interfaces.
#
#   make                    build ./cavity against the MeepHoltz install in MEEP_PREFIX
#   make check              tiny serial + MPI smoke test of every iface/solver combination
#   make convergence        local grid refinement over RES_LIST, then analyze.py
#   make strong|weak|conv   submit the TinkerCliff scaling / convergence jobs (submit.sh)
#   make analyze            summarize results/ into results/summary.csv (+ plots)
#   make clean
#
# solve_wh settings (lists allowed; every combination runs, cw once):
#   WHSOLVER=gmres|bicgstab   PERIODS=1   e.g.  make convergence WHSOLVER="gmres bicgstab" PERIODS="1 2"
# Source angular frequency: OMEGA [1]; a non-default one writes to results/local_w<OMEGA>.
# The same variables drive cavity.slurm / submit.sh.
#
# Locally MEEP_ROOT is the MeepHoltz checkout (install in _local, built by build_meep.sh).
# On TinkerCliff, `source env_tinkercliff.sh` first; it sets MEEP_ROOT/MEEP_PREFIX/PYTHON.
#
# libmeep is built with mpicxx (it references MPI_Init), so an MPI compiler wrapper is required.

MEEP_ROOT   ?= $(HOME)/Develop/MeepHoltz
MEEP_PREFIX ?= $(MEEP_ROOT)/_local
PYTHON      ?= $(MEEP_ROOT)/venv/bin/python
MPIRUN      ?= mpirun
NP          ?= 2
RES_LIST    ?= 8 12 16 24 32
WHSOLVER    ?= gmres
PERIODS     ?= 1
OMEGA       ?= 1
export WHSOLVER PERIODS OMEGA
LOCAL_OUT   := results/local$(if $(filter-out 1,$(OMEGA)),_w$(OMEGA))

# "cw" plus "wh -whsolver W -periods P" for every W x P (spaces kept as _ inside one word)
WH_VARIANTS := $(foreach w,$(WHSOLVER),$(foreach p,$(PERIODS),wh_-whsolver_$(w)_-periods_$(p)))
VARIANTS    := $(WH_VARIANTS) cw

# `CXX ?=` would not take effect: make defines CXX itself.
ifeq ($(origin CXX),default)
  CXX := mpicxx
endif
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -W

# meep.pc lists libmeep's own dependencies (hdf5, gsl, fftw, ...) for static builds too.
PKG_CONFIG := PKG_CONFIG_PATH=$(MEEP_PREFIX)/lib/pkgconfig:$(MEEP_PREFIX)/lib64/pkgconfig$(if $(PKG_CONFIG_PATH),:$(PKG_CONFIG_PATH)) pkg-config
CPPFLAGS   += -I$(MEEP_PREFIX)/include $(shell $(PKG_CONFIG) --cflags meep 2>/dev/null)
LDLIBS     += $(or $(shell $(PKG_CONFIG) --libs meep 2>/dev/null),-L$(MEEP_PREFIX)/lib -lmeep)
LDFLAGS    += -Wl,-rpath,$(MEEP_PREFIX)/lib

# Python package of the same install, in case it is not on the venv's path already
MEEP_PYPATH := $(firstword $(wildcard $(MEEP_PREFIX)/lib/python3*/site-packages))
export PYTHONPATH := $(MEEP_PYPATH)$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: all check convergence strong weak conv analyze clean

all: cavity

cavity: cavity.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

check: cavity
	@for v in wh_-whsolver_gmres_-periods_1 wh_-whsolver_bicgstab_-periods_2 cw; do \
	  s=$$(echo $$v | tr _ ' '); \
	  ./cavity -solver $$s -res 8 | grep RESULT || exit 1; \
	  $(MPIRUN) -n $(NP) ./cavity -solver $$s -res 8 | grep RESULT || exit 1; \
	  $(PYTHON) cavity.py -solver $$s -res 8 | grep RESULT || exit 1; \
	  $(MPIRUN) -n $(NP) $(PYTHON) cavity.py -solver $$s -res 8 | grep RESULT || exit 1; \
	done

convergence: cavity
	@mkdir -p $(LOCAL_OUT)
	@for r in $(RES_LIST); do for v in $(VARIANTS); do \
	  s=$$(echo $$v | tr _ ' '); t=$$(echo $$v | sed 's/_-whsolver_/-/; s/_-periods_/-p/'); \
	  $(MPIRUN) -n $(NP) ./cavity -solver $$s -res $$r -omega $(OMEGA) > $(LOCAL_OUT)/conv_cpp_$${t}_np$(NP)_r$$r.log 2>&1; \
	  $(MPIRUN) -n $(NP) $(PYTHON) cavity.py -solver $$s -res $$r -omega $(OMEGA) > $(LOCAL_OUT)/conv_py_$${t}_np$(NP)_r$$r.log 2>&1; \
	  grep -h RESULT $(LOCAL_OUT)/conv_*_$${t}_np$(NP)_r$$r.log; \
	done; done
	$(PYTHON) analyze.py $(LOCAL_OUT)

strong weak conv: cavity
	./submit.sh $@

analyze:
	$(PYTHON) analyze.py results

clean:
	rm -rf cavity cavity.dSYM

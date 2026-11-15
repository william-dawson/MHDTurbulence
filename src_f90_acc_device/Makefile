exed := ../exe

exe=Simulation.x

all: ${exed}/${exe}

.PHONY: all clean

#################
# Fortran
#################

fc= mpifort
fopt = -g -traceback -O2
foptopenacc = -Minfo=accel -acc -mcmodel=medium -gpu=maxregcount:160
# UpdateConsv gets its own register cap, independent of the flux
# kernels' 160 -- set via UC_MAXREGCOUNT on the make command line, e.g.
# `make UC_MAXREGCOUNT=64`. Empty means no cap (use the compiler default).
UC_MAXREGCOUNT ?= 32
ifeq ($(strip $(UC_MAXREGCOUNT)),)
foptopenacc_uc = -Minfo=accel -acc -mcmodel=medium -gpu=ptxinfo
else
foptopenacc_uc = -Minfo=accel -acc -mcmodel=medium -gpu=maxregcount:$(UC_MAXREGCOUNT),ptxinfo
endif

# CUDA kernel arms (FugakuNEXT-cuda): the five core kernels compile as
# CUDA C++ (mhd_cuda.cu, nvc++) and launch on the OpenACC stream, while
# every other region stays OpenACC. `make CUDA_KERNELS=1` enables all
# three groups (sv = StateVevtor, uc = UpdateConsv, flux = NumericalFlux
# 1/2/3); a list enables subsets, e.g. `make CUDA_KERNELS=sv,uc`.
# mhd.f90 and updateconsv.f90 are preprocessed (-Mpreprocess) in every
# build so the #ifdef guards resolve; with no arms the OpenACC branch
# compiles, i.e. the default build is unchanged.
CUDA_KERNELS ?=
FLUX_BLOCK ?= 128
FLUX_MINBLOCKS ?= 4
FACES_PER_THREAD ?= 1
UC_X ?= 32
UC_Y ?= 8
UC_MINBLOCKS ?= 5
SV_X ?= 32
SV_Y ?= 16
SV_MINBLOCKS ?= 4
ifeq ($(CUDA_KERNELS),1)
CUDA_ARMS := sv uc flux
else
CUDA_ARMS := $(CUDA_KERNELS)
endif
CPPDEFS :=
ifneq ($(filter sv,$(CUDA_ARMS)),)
CPPDEFS += -DUSE_CUDA_SV
endif
ifneq ($(filter uc,$(CUDA_ARMS)),)
CPPDEFS += -DUSE_CUDA_UC
endif
ifneq ($(filter flux,$(CUDA_ARMS)),)
CPPDEFS += -DUSE_CUDA_FLUX
endif
ifneq ($(strip $(CPPDEFS)),)
CPPDEFS += -DUSE_CUDA_DATA_MAP
endif
ifneq ($(strip $(CPPDEFS)),)
CUDA_OBJ := mhd_cuda.o
# CUDA_SM: target architecture for the CUDA kernels (sm_100 on B200/Rikyu,
# sm_90 on H100/Miyabi-G). The runtime lives under the nvcc-found CUDA
# toolkit: <nvcc bindir>/../../cuda/<version>/lib64 on both machines.
CUDA_SM ?= sm_100
CUDA_VERSION := $(shell nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
CUDA_LIBDIR  := $(shell ls -d $$(dirname $$(which nvcc))/../../cuda/$(CUDA_VERSION)/lib64)
LDLIBS += -L$(CUDA_LIBDIR) -lcudart -lstdc++
endif

.SUFFIXES: .f90 .o

.f90.o:
	${fc} ${fopt} ${foptopenacc} -c $<

OBJS = main.o output.o boundary.o mhd.o mpiroutines.o config.o updateconsv.o gravforce.o $(CUDA_OBJ)

config.o: config.f90
mpiroutines.o: mpiroutines.f90 config.o
mhd.o: mhd.f90 mpiroutines.o config.o
	${fc} ${fopt} ${foptopenacc} -Mpreprocess ${CPPDEFS} -c $<
updateconsv.o: updateconsv.f90 mhd.o mpiroutines.o config.o
	${fc} ${fopt} ${foptopenacc_uc} -Mpreprocess ${CPPDEFS} -c $<
# GravForce lives in its own unit: inserting the cuda_kernels_f90 glue
# module into mhd.f90 changed nvfortran's codegen for this loop and
# re-triggered the unsound-LDG.128 misalignment fault (same class as
# the odd-in pad workaround) at the first GravForce region.
gravforce.o: gravforce.f90 mhd.o mpiroutines.o config.o
	${fc} ${fopt} ${foptopenacc} -c $<
mhd_cuda.o: mhd_cuda.cu
	nvcc -O3 -lineinfo -gencode arch=$(subst sm_,compute_,$(CUDA_SM)),code=$(CUDA_SM) \
		-DFLUX_BLOCK=$(FLUX_BLOCK) -DFLUX_MINBLOCKS=$(FLUX_MINBLOCKS) \
		-DFACES_PER_THREAD=$(FACES_PER_THREAD) \
		-DUC_X=$(UC_X) -DUC_Y=$(UC_Y) -DUC_MINBLOCKS=$(UC_MINBLOCKS) \
		-DSV_X=$(SV_X) -DSV_Y=$(SV_Y) -DSV_MINBLOCKS=$(SV_MINBLOCKS) -c $< -o $@
boundary.o: boundary.f90 mhd.o mpiroutines.o config.o
output.o: output.f90 boundary.o mhd.o mpiroutines.o
main.o: main.f90 output.o boundary.o mhd.o mpiroutines.o updateconsv.o
	${fc} ${fopt} ${foptopenacc} -Mpreprocess ${CPPDEFS} -c $<

${exed}/${exe}: ${OBJS}
	${fc} ${fopt} ${foptopenacc} ${OBJS} -o $@ ${LDLIBS}

#################
# clean up
#################
clean:
	rm -f ${exed}/${exe} *.o *.mod *~

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

#fopt = -g -traceback -O2
#fopt = -g -traceback -check all -fpe0

.SUFFIXES: .f90 .o

.f90.o:
	${fc} ${fopt} ${foptopenacc} -c $<

OBJS = main.o output.o boundary.o mhd.o mpiroutines.o config.o updateconsv.o

config.o: config.f90
mpiroutines.o: mpiroutines.f90 config.o
mhd.o: mhd.f90 mpiroutines.o config.o
updateconsv.o: updateconsv.f90 mhd.o mpiroutines.o config.o
	${fc} ${fopt} ${foptopenacc_uc} -c $<
boundary.o: boundary.f90 mhd.o mpiroutines.o config.o
output.o: output.f90 boundary.o mhd.o mpiroutines.o
main.o: main.f90 output.o boundary.o mhd.o mpiroutines.o updateconsv.o

${exed}/${exe}: ${OBJS}
	${fc} ${fopt} ${foptopenacc} ${OBJS} -o $@

#################
# clean up
#################
clean:
	rm -f ${exed}/${exe} *.o *.mod *~

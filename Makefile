CC      = nvcc
VECTOR  = -Xcompiler -ffast-math -Xcompiler -march=native -Xcompiler -ftree-vectorize
OPENMP  = -Xcompiler -fopenmp
CFLAGS	= -Xcompiler -O3 -Xcompiler -funroll-loops $(VECTOR) $(OPENMP) $(DEFINE)
#CFLAGS	= -O3 -flto -funroll-loops $(VECTOR) $(OPENMP) $(DEFINE)
WFLAGS	= -Xcompiler -std=c11 -Xcompiler -Wall -Xcompiler -Wextra -g
LDFLAGS	= -lm

TARGETS	= tiny_md viz
SOURCES	= $(shell echo *.c)
OBJECTS = core.o wtime.o

all: $(TARGETS)

viz: viz.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lGL -lGLU -lglut

tiny_md: tiny_md.o $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	
tiny_md_omp: tiny_md_omp.o $(OBJECTS_OMP)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

core.o: core.cu
	$(CC) $(WFLAGS) $(CPPFLAGS) $(CFLAGS) -c $<

%.o: %.c
	$(CC) $(WFLAGS) $(CPPFLAGS) $(CFLAGS) -c $<

clean:
	rm -f $(TARGETS) *.o *.xyz *.log .depend

.depend: $(SOURCES)
	$(CC) -MM $^ > $@

-include .depend

.PHONY: clean all

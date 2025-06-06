# Compilers
CC      ?= gcc
MPICC   ?= mpicc

# Flags
CFLAGS  ?= -Wall -fopenmp-extensions -g
LDFLAGS ?= -lm
MPIFLAGS ?= -np 4 --oversubscribe

# Folder for benchmark input files
FOLDER  ?= ./res

# Targets
all: wordfreq_omp wordfreq_mpi

wordfreq_omp: wordfreq_omp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

wordfreq_mpi: wordfreq_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f wordfreq_omp wordfreq_mpi

benchmark-omp: wordfreq_omp
	./wordfreq_omp -b -n 8 $(FOLDER)/*.txt

benchmark-mpi: wordfreq_mpi
	mpiexec $(MPIFLAGS) ./wordfreq_mpi $(FOLDER)/*.txt

.PHONY: all clean benchmark-omp benchmark-mpi

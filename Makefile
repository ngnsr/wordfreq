CC = /usr/local/opt/gcc/bin/gcc-14
CFLAGS = -Wall -fopenmp -g
LDFLAGS = -lm
MPIFLAGS = -np 4 --oversubscribe

all: wordfreq_omp wordfreq_mpi 

wordfreq_omp: wordfreq_omp.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

wordfreq_mpi: wordfreq_mpi.c
	mpicc -o wordfreq_mpi wordfreq_mpi.c

clean:
	rm -f wordfreq_omp wordfreq_mpi

benchmark-omp: all 
	./wordfreq_omp -b -n 8 ./res/*.txt

benchmark-mpi: all 
	mpirun ${MPIFLAGS} ./wordfreq_mpi ./res/*.txt

.PHONY: all clean benchmark

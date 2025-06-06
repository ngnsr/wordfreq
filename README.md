# wordfreq

**wordfreq** is a high-performance command-line tool implemented in C for analyzing word frequencies in large text files. It supports parallel processing using OpenMP and MPI for efficient computation on shared-memory and distributed systems.

## 🚀 Features

- Word frequency analysis on large text files.
- Parallel execution using OpenMP and MPI.
- Configurable number of processes and input folder.
- Easy build with `make`.

## 🛠️ Installation & Build

### Prerequisites

- **C Compiler** supporting OpenMP (GCC recommended).
- **MPI library** (e.g., OpenMPI or MPICH).
- `make` utility.

### Build instructions

Clone the repository:

```bash
git clone https://github.com/ngnsr/wordfreq.git
cd wordfreq
```

Build both OpenMP and MPI versions:

```bash
make all
```

### Customizing build

You can override compilers and flags:

```bash
make CC=gcc-15 MPICC=mpicc-15
```

## 📁 Usage

### Benchmark with OpenMP version

```bash
make benchmark-omp
```

Specify input folder and number of threads:

```bash
make benchmark-omp FOLDER=./test_files CFLAGS="-Wall -fopenmp -g"
```

### Benchmark with MPI version

```bash
make benchmark-mpi
```

Override MPI flags and input folder:

```bash
make benchmark-mpi FOLDER=./test_files MPIFLAGS="-np 8 --oversubscribe"
```

Alternatively, run directly:

```bash
mpiexec -np 4 ./wordfreq_mpi ./res/*.txt
```

## 🧹 Clean build files

```bash
make clean
```

## 📂 Directory structure

```
wordfreq/
├── wordfreq_omp.c       # OpenMP implementation
├── wordfreq_mpi.c       # MPI implementation
├── Makefile             # Build script
└── res/                 # Default folder with input text files
```

_Note:_ The `-fopenmp-extensions` flag is GCC-specific and may not be supported on all platforms or compilers. If you encounter build issues, try removing it from `CFLAGS`.

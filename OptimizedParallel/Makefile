CC = mpicc
NVCC = nvcc
CUDA_PATH = /usr/local/cuda

CFLAGS = -I$(CUDA_PATH)/include
LFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lm

all: program

program: main.o matrix_operations.o
	$(CC) -o program main.o matrix_operations.o $(LFLAGS)

main.o: main.c
	$(CC) -c main.c $(CFLAGS)

matrix_operations.o: matrix_operations.cu
	$(NVCC) -c matrix_operations.cu $(CFLAGS)

clean:
	rm -f *.o program

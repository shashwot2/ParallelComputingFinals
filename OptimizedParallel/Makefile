nvcc -c -o matrix_operations.o matrix_operations.cu
mpicc -c -o main.o main.c
mpicc -o program main.o matrix_operations.o -L/usr/local/cuda/lib64 -lcudart

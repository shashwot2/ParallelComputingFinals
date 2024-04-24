all: matrixmultMPI.c matrixmultCUDA.cu
	mpicc -O3 matrixmultMPI.c -c -o highlifeMPI.o
	nvcc -O3 matrixmultCUDA.cu -c -o matrixmultCUDA.o 
	mpicc -O3 highlifeMPI.o matrixmultCUDA.o -o matrixmult-MPI-CUDA-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ -lm 
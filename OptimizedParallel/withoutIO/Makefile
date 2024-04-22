all: highlifeMPI-CUDA.c highlifeCUDA.cu
	mpicc -g highlifeMPI-CUDA.c -c -o highlifeMPI-CUDA.o
	nvcc -g -G highlifeCUDA.cu -c -o highlifeCUDA.o 
	mpicc -g highlifeMPI-CUDA.o highlifeCUDA.o -o highlife-MPI-CUDA-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ -lm 
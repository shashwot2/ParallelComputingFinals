all: highlifeMPI-CUDA.c highlifeCUDA.cu
	mpicc -O3 highlifeMPI-CUDA.c -c -o highlifeMPI-CUDA.o
	nvcc -O3 highlifeCUDA.cu -c -o highlifeCUDA.o 
	mpicc -O3 highlifeMPI-CUDA.o highlifeCUDA.o -o highlife-MPI-CUDA-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ -lm 
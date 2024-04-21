#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMulKernel(int *a, int* b, int *c, int width, int local_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < local_width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; k++) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

extern "C" void matrixMulCUDA(int *h_a, int* h_b, int *h_c, int width, int local_width) {
    int *d_a, *d_b, *d_c;
    cudaError_t err;

    err = cudaMalloc((void **)&d_a, local_width * width * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_b, width * width * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_c, local_width * width * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_a, h_a, local_width * width * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_b, h_b, width * width * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (local_width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, width, local_width);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch matrix multiplication kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_c, d_c, local_width * width * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
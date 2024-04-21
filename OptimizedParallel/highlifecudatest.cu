#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMultiplyNaive(int *a, int* b, int *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; k++) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

extern "C" void performMatrixMultiplication(int *h_A, int* h_B, int *h_C, int local_rows, int N) {
    int *d_A, *d_B, *d_C;
    size_t size_A = local_rows * N * sizeof(int);
    size_t size_B = N * N * sizeof(int);
    size_t size_C = local_rows * N * sizeof(int);

    cudaError_t err;

    err = cudaMalloc((void **)&d_A, size_A);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc A Error: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc((void **)&d_B, size_B);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc B Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return;
    }

    err = cudaMalloc((void **)&d_C, size_C);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc C Error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (local_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplyNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
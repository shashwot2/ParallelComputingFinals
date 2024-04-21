#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMulKernel(int *a, int *b, int *c, int width, int local_width) {
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

extern "C" void matrixMulCUDA(int *h_a, int *h_b, int *h_c, int width, int local_width) {
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, local_width * width * sizeof(int));
    cudaMalloc((void **)&d_b, width * width * sizeof(int));
    cudaMalloc((void **)&d_c, local_width * width * sizeof(int));

    cudaMemcpy(d_a, h_a, local_width * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, width * width * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (local_width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, width, local_width);

    cudaMemcpy(h_c, d_c, local_width * width * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
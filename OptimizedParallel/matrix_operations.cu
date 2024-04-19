#include <cuda_runtime.h>

__global__ void matrixMultiplyNaive(int *A, int *B, int *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (row < n && col < n)
    {
        for (int k = 0; k < n; k++)
        {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

extern "C" void performMatrixMultiplication(int *h_A, int *h_B, int *h_C, int N, int threadsPerBlock)
{
    int *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(int);
    int rows_per_process = N / size;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (rows_per_process + threads.y - 1) / threads.y);


    matrixMultiplyNaive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
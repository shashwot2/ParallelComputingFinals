#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>

extern void performMatrixMultiplication(int *A, int *B, int *C, int N);

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 8;  
    int *a, *b, *c;
    a = (int *)malloc(N * N * sizeof(int));
    b = (int *)malloc(N * N * sizeof(int));
    c = (int *)malloc(N * N * sizeof(int));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = 5; 
            b[i * N + j] = 5;
        }
    }

    performMatrixMultiplication(a, b, c, N);

    if (rank == 0) {
        printf("Result Matrix C:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", c[i * N + j]);
            }
            printf("\n");
        }
    }

    free(a);
    free(b);
    free(c);

    MPI_Finalize();
    return 0;
}
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

extern void performMatrixMultiplication(int *h_A, int *h_B, int *h_C, int N, int threadsPerBlock);
void printMatrix(const char *name, int *matrix, int rows, int cols)
{
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%5d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Usage: %s <matrix_size> <threads_per_block>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);

    if (N % size != 0)
    {
        if (rank == 0)
        {
            fprintf(stderr, "Matrix size %d must be divisible by number of processes %d.\n", N, size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *A, *B, *C;
    int *subA, *subC;
    int rows_per_process = N / size;

    if (rank == 0)
    {
        A = (int *)malloc(N * N * sizeof(int));
        B = (int *)malloc(N * N * sizeof(int));
        C = (int *)malloc(N * N * sizeof(int));

        for (int i = 0; i < N * N; i++)
        {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
    } else {
    B = (int *)malloc(N * N * sizeof(int));
}

    subA = (int *)malloc(rows_per_process * N * sizeof(int));
    subC = (int *)malloc(rows_per_process * N * sizeof(int));

    MPI_Scatter(A, rows_per_process * N, MPI_INT, subA, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    performMatrixMultiplication(subA, B, subC, N, threadsPerBlock);
    MPI_Gather(subC, rows_per_process * N, MPI_INT, C, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        if (N < 10)
        {
            printMatrix("Matrix A", A, N, N);
            printMatrix("Matrix B", B, N, N);
            printMatrix("Matrix C", C, N, N);
        }
        free(A);
        free(B);
        free(C);
    }

    free(subA);
    free(subC);

    MPI_Finalize();
    return 0;
}

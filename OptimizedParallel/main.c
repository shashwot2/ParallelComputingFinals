#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

extern void performMatrixMultiplication(int *h_A, int *h_B, int *h_C, int N);
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
    rank = 2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 4;
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
            A[i] = rand() % 100;
            B[i] = rand() % 100;
        }
    }

    subA = (int *)malloc(rows_per_process * N * sizeof(int));
    subC = (int *)malloc(rows_per_process * N * sizeof(int));

    MPI_Scatter(A, rows_per_process * N, MPI_INT, subA, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    performMatrixMultiplication(subA, B, subC, N);

    MPI_Gather(subC, rows_per_process * N, MPI_INT, C, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printMatrix("Matrix A", A, N, N);
        printMatrix("Matrix B", B, N, N);
        printMatrix("Matrix C", C, N, N);

        free(A);
        free(B);
        free(C);
    }

    free(subA);
    free(subC);

    MPI_Finalize();
    return 0;
}

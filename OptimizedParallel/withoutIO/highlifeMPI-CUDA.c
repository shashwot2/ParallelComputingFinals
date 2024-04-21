#define N 4
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stddef.h>
#include "mpi.h"

extern void matrixMulCUDA(int *a, int *b, int *c, int width, int local_width);
void print_results(char *prompt, int a[N][N]);

int main(int argc, char *argv[])
{
    int i, j, rank, size, tag = 99;
    int a[N][N]={{1,2,3,4},{5,6,7,8},{9,1,2,3},{4,5,6,7}};
    int b[N][N]={{1,2,3,4},{5,6,7,8},{9,1,2,3},{4,5,6,7}};
    int c[N][N];
    int aa[N*N/4], cc[N*N/4];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Scatter rows of first matrix to different processes
    MPI_Scatter(a, N*N/size, MPI_INT, aa, N*N/size, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast second matrix to all processes
    MPI_Bcast(b, N*N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Perform matrix multiplication using CUDA kernel
    matrixMulCUDA(aa, (int *)b, cc, N, N/size);

    MPI_Gather(cc, N*N/size, MPI_INT, c, N*N/size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    if (rank == 0) {
        print_results("C = ", c);
    }

    return 0;
}

void print_results(char *prompt, int a[N][N])
{
    int i, j;

    printf ("\n\n%s\n", prompt);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf(" %d", a[i][j]);
        }
        printf ("\n");
    }
    printf ("\n\n");
}
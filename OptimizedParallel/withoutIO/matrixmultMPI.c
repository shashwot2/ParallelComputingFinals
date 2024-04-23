#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdbool.h>
#include <math.h>

extern void matrixMulCUDA(int *a, int *b, int *c, int width, int local_width, int threads);
void print_results(char *prompt, int *a, int size);

int main(int argc, char *argv[])
{   
    bool isWeakScaling = false;
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matrix size> <threads >\n", argv[0]);
        exit(1);
    }
    
    int n = atoi(argv[1]); // Dynamic matrix size
    int threads = atoi(argv[2]); // Dynamic threads
    int i, j, rank, size, tag = 99;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int *a, *b, *c, *aa, *cc;
    if (isWeakScaling) {
        n = n * (int)sqrt(size);
    } 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Allocate memory
    a = (int *)malloc(n * n * sizeof(int));
    b = (int *)malloc(n * n * sizeof(int));
    c = (int *)malloc(n * n * sizeof(int));
    aa = (int *)malloc(n * n / size * sizeof(int));
    cc = (int *)malloc(n * n / size * sizeof(int));

    // Initialize matrices
    if (rank == 0) {
        for (i = 0; i < n * n; i++) {
            a[i] = rand() % 100; // Dynamic content
            b[i] = rand() % 100;
        }
    }

    // Scatter rows of first matrix to different processes
    MPI_Scatter(a, n*n/size, MPI_INT, aa, n*n/size, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast second matrix to all processes
    MPI_Bcast(b, n*n, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform matrix multiplication using CUDA kernel
    matrixMulCUDA(aa, b, cc, n, n/size, threads);

    // Gather results from all processes
    MPI_Gather(cc, n*n/size, MPI_INT, c, n*n/size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0 && n < 12) {
        print_results("A = ", a, n);
        print_results("B = ", b, n);
        print_results("C = ", c, n);
    }

    // Cleanup
    free(a); free(b); free(c); free(aa); free(cc);
    MPI_Finalize();

    return 0;
}

void print_results(char *prompt, int *a, int size)
{
    int i, j;

    printf ("\n\n%s\n", prompt);
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            printf(" %d", a[i * size + j]);
        }
        printf ("\n");
    }
    printf ("\n\n");
}

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
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <matrix size> <threads >\n", argv[0]);
        exit(1);
    }

    int n = atoi(argv[1]);       // Dynamic matrix size
    int threads = atoi(argv[2]); // Dynamic threads
    int i, j, rank, size, tag = 99;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int *a, *b, *c, *aa, *cc;
    if (isWeakScaling)
    {
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
    if (rank == 0)
    {
        for (i = 0; i < n * n; i++)
        {
            a[i] = rand() % 100; // Dynamic content
            b[i] = rand() % 100;
        }
    }

    uint64_t total_start_time = clock_now();

    uint64_t scatter_start_time = clock_now();
    MPI_Scatter(a, n * n / size, MPI_INT, aa, n * n / size, MPI_INT, 0, MPI_COMM_WORLD);
    uint64_t scatter_end_time = clock_now();

    uint64_t bcast_start_time = clock_now();
    MPI_Bcast(b, n * n, MPI_INT, 0, MPI_COMM_WORLD);
    uint64_t bcast_end_time = clock_now();

    uint64_t compute_start_time = clock_now();
    matrixMulCUDA(aa, b, cc, n, n / size, threads);
    uint64_t compute_end_time = clock_now();

    uint64_t gather_start_time = clock_now();
    MPI_Gather(cc, n * n / size, MPI_INT, c, n * n / size, MPI_INT, 0, MPI_COMM_WORLD);
    uint64_t gather_end_time = clock_now();

    uint64_t total_end_time = clock_now();

    if (rank == 0 && n < 12)
    {
        print_results("A = ", a, n);
        print_results("B = ", b, n);
        print_results("C = ", c, n);
    }
    if (rank == 0)
    {
        double scatter_time = (double)(scatter_end_time - scatter_start_time) / 512000000.0;
        double bcast_time = (double)(bcast_end_time - bcast_start_time) / 512000000.0;
        double compute_time = (double)(compute_end_time - compute_start_time) / 512000000.0;
        double gather_time = (double)(gather_end_time - gather_start_time) / 512000000.0;
        double total_time = (double)(total_end_time - total_start_time) / 512000000.0;

        printf("Scatter time: %.6f seconds\n", scatter_time);
        printf("Broadcast time: %.6f seconds\n", bcast_time);
        printf("Compute time: %.6f seconds\n", compute_time);
        printf("Gather time: %.6f seconds\n", gather_time);
        printf("Total execution time: %.6f seconds\n", total_time);
    }
    // Cleanup
    free(a);
    free(b);
    free(c);
    free(aa);
    free(cc);
    MPI_Finalize();

    return 0;
}

void print_results(char *prompt, int *a, int size)
{
    int i, j;

    printf("\n\n%s\n", prompt);
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            printf(" %d", a[i * size + j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

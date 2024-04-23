#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void print_results(char *prompt, int *a, int rows, int cols);
extern void matrixMulCUDA(int *a, int *b, int *c, int width, int local_width, int threads);

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);

    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <matrixsize_per_process> <threads>\\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N_per_process = atoi(argv[1]);
    int threads = atoi(argv[2]);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = N_per_process * size;



    int *aa = (int *)calloc(N_per_process * N, sizeof(int));
    int *bb = (int *)calloc(N * N, sizeof(int));
    int *cc = (int *)calloc(N_per_process * N, sizeof(int));

    MPI_File fh_a, fh_b, fh_c;
    MPI_Status status;

    // Read matrix A
    if (MPI_File_open(MPI_COMM_WORLD, "matrix_a.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_a) != MPI_SUCCESS)
    {
        fprintf(stderr, "Failed to open file 'matrix_a.dat'\\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_File_read_at_all(fh_a, rank * N_per_process * N * sizeof(int), aa, N_per_process * N, MPI_INT, &status);
    MPI_File_close(&fh_a);

    // Read matrix B
    if (MPI_File_open(MPI_COMM_WORLD, "matrix_b.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_b) != MPI_SUCCESS)
    {
        fprintf(stderr, "Failed to open file 'matrix_b.dat'\\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_File_read_at_all(fh_b, 0, bb, N * N, MPI_INT, &status);
    MPI_File_close(&fh_b);

    // Process local matrix multiplication
    matrixMulCUDA(aa, bb, cc, N, N_per_process, threads);

    // Printing results
    if (N <= 12)
    {
        char prompt[100];
        sprintf(prompt, "Result Matrix on Process %d:", rank);
        print_results(prompt, cc, N_per_process, N);
    }

    MPI_File_open(MPI_COMM_WORLD, "result_matrix.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_c);
    MPI_File_write_at_all(fh_c, rank * N_per_process * N * sizeof(int), cc, N_per_process * N, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh_c);

    // Cleanup
    free(aa);
    free(bb);
    free(cc);

    MPI_Finalize();
    return 0;
}

void print_results(char *prompt, int *a, int rows, int cols)
{
    printf("%s\\n", prompt);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", a[i * cols + j]);
        }
        printf("\\n");
    }
    printf("\\n");
}
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int N= 4;
void print_results(char *prompt, int *a, int rows, int cols);
extern void matrixMulCUDA(int *a, int *b, int *c, int width, int local_width, int threads);

int main(int argc, char *argv[])
{
    // IMPORTANT PLEASE READ: The matrix size is hardcoded here
    // Matrix size actually shouldn't be defined here in input withIO because the size works from reading the input file directly
    int rank, size;
    MPI_Init(&argc, &argv);
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <matrixsize> <threads>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    N = atoi(argv[1]);
    int threads = atoi(argv[2]);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = N / size;
    int remaining_rows = N % size;

    int *send_counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        send_counts[i] = rows_per_process * N;
        displs[i] = i * rows_per_process * N;
        if (i == size - 1)
        {
            send_counts[i] += remaining_rows * N; 
        }
    }

    int *aa = (int *)calloc(send_counts[rank], sizeof(int));
    int *bb = (int *)calloc(N * N, sizeof(int));
    int *cc = (int *)calloc(send_counts[rank], sizeof(int));

    MPI_File fh_a, fh_b, fh_c;
    MPI_Status status;

    // Read matrix A
    if (MPI_File_open(MPI_COMM_WORLD, "matrix_a.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_a) != MPI_SUCCESS)
    {
        fprintf(stderr, "Failed to open file 'matrix_a.txt'\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_File_read_at_all(fh_a, displs[rank] * sizeof(int), aa, send_counts[rank], MPI_INT, &status);
    MPI_File_close(&fh_a);

    // Read matrix B
    if (MPI_File_open(MPI_COMM_WORLD, "matrix_b.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_b) != MPI_SUCCESS)
    {
        fprintf(stderr, "Failed to open file 'matrix_b.txt'\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_File_read_at_all(fh_b, 0, bb, N * N, MPI_INT, &status);
    MPI_File_close(&fh_b);

    // Process local matrix multiplication
    matrixMulCUDA(aa, bb, cc, N, send_counts[rank] / N);

    // Printing results
    if (N <= 12)
    {
    print_results(prompt, cc, send_counts[rank] / N, N);
    char prompt[100];
    sprintf(prompt, "Result Matrix on Process %d:", rank);
    }
    MPI_File_open(MPI_COMM_WORLD, "result_matrix.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_c);
    MPI_File_write_at_all(fh_c, displs[rank] * sizeof(int), cc, send_counts[rank], MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh_c);

    // Cleanup
    free(aa);
    free(bb);
    free(cc);
    free(send_counts);
    free(displs);

    MPI_Finalize();
    return 0;
}

void print_results(char *prompt, int *a, int rows, int cols)
{
    printf("%s\n", prompt);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ", a[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

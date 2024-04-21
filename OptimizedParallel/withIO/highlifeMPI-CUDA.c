#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>

extern void performMatrixMultiplication(int *A, int *B, int *C, int N);

int N = 4; // Assuming a fixed size for simplicity, adjust as needed.

void handle_error(int errcode, const char *str);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_rank = N / size;
    int *a = malloc(N * N * sizeof(int));
    int *b = malloc(N * N * sizeof(int));
    int *c = malloc(N * N * sizeof(int));
    int *gathered_c = NULL;

    if (rank == 0)
    {
        gathered_c = malloc(N * N * sizeof(int));
    }

    MPI_File fh_a, fh_b, fh_c;
    MPI_Status status;

    // Read matrix A and B from files
    MPI_File_open(MPI_COMM_WORLD, "matrix_a.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_a);
    MPI_File_open(MPI_COMM_WORLD, "matrix_b.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_b);
    printf("Rank %d: Offset for Matrix B: %d\n", rank, rank * rows_per_rank * N * sizeof(int));

    MPI_File_read_at(fh_a, rank * rows_per_rank * N * sizeof(int), a, rows_per_rank * N, MPI_INT, &status);
    MPI_File_read_at(fh_b, rank * rows_per_rank * N * sizeof(int), b, rows_per_rank * N, MPI_INT, &status);

    MPI_File_close(&fh_a);
    MPI_File_close(&fh_b);

    // Matrix multiplication
    performMatrixMultiplication(a, b, c, N);

    // Gather results at root
    MPI_Gather(c, rows_per_rank * N, MPI_INT, gathered_c, rows_per_rank * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Complete Matrix C gathered at root:\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                printf("%d ", gathered_c[i * N + j]);
            }
            printf("\n");
        }
    }

    MPI_File_open(MPI_COMM_WORLD, "matrix_c.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_c);
    MPI_File_write_at(fh_c, rank * rows_per_rank * N * sizeof(int), c, rows_per_rank * N, MPI_INT, &status);
    MPI_File_close(&fh_c);

    free(a);
    free(b);
    free(c);
    if (rank == 0)
    {
        free(gathered_c);
    }

    MPI_Finalize();
    return 0;
}

void handle_error(int errcode, const char *str)
{
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    fprintf(stderr, "%s: %s\n", str, msg);
    MPI_Abort(MPI_COMM_WORLD, errcode);
}

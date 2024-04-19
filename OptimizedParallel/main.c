#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

extern void performMatrixMultiplication(int *subA, int *B, int *subC, int N, int threadsPerBlock);

void readMatrixMPIIO(const char *filename, int *matrix, int startRow, int numRows, int totalSize, MPI_Comm comm) {
    MPI_File fh;
    MPI_Status status;
    MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_seek(fh, sizeof(int) * startRow * totalSize, MPI_SEEK_SET);
    MPI_File_read(fh, matrix, numRows * totalSize, MPI_INT, &status);
    MPI_File_close(&fh);
}

void writeMatrixMPIIO(const char *filename, int *matrix, int startRow, int numRows, int totalSize, MPI_Comm comm) {
    MPI_File fh;
    MPI_Status status;
    MPI_File_open(comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_seek(fh, sizeof(int) * startRow * totalSize, MPI_SEEK_SET);
    MPI_File_write(fh, matrix, numRows * totalSize, MPI_INT, &status);
    MPI_File_close(&fh);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <matrix_size> <threads_per_block> <file_prefix>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    char *file_prefix = argv[3];

    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Matrix size %d must be divisible by number of processes %d.\n", N, size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_per_process = N / size;
    int *subA = (int *)malloc(rows_per_process * N * sizeof(int));
    int *B = (int *)malloc(N * N * sizeof(int));
    int *subC = (int *)malloc(rows_per_process * N * sizeof(int));

    char filenameA[100], filenameB[100], filenameC[100];
    sprintf(filenameA, "%s_A.bin", file_prefix);
    sprintf(filenameB, "%s_B.bin", file_prefix);
    sprintf(filenameC, "%s_C.bin", file_prefix);

    readMatrixMPIIO(filenameA, subA, rank * rows_per_process, rows_per_process, N, MPI_COMM_WORLD);
    readMatrixMPIIO(filenameB, B, 0, N, N, MPI_COMM_WORLD); 

    performMatrixMultiplication(subA, B, subC, N, threadsPerBlock);

    writeMatrixMPIIO(filenameC, subC, rank * rows_per_process, rows_per_process, N, MPI_COMM_WORLD);

    free(subA);
    free(B);
    free(subC);

    MPI_Finalize();
    return 0;
}

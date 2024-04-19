#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern void cudaMatrixMultiply(int *subA, int *B, int *subC, int n, int size, int rank);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1024; 
    int rows = n / size; 

    int *subA, *B, *subC;
    cudaMallocManaged(&subA, rows * n * sizeof(int));
    cudaMallocManaged(&B, n * n * sizeof(int));
    cudaMallocManaged(&subC, rows * n * sizeof(int));

    MPI_File fh_A, fh_B, fh_C;
    MPI_Status status;

    MPI_File_open(MPI_COMM_WORLD, "matrix_A.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_A);
    MPI_File_open(MPI_COMM_WORLD, "matrix_B.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_B);

    MPI_Offset offset_A = rank * rows * n * sizeof(int);
    MPI_Offset offset_B = 0;

    MPI_File_read_at(fh_A, offset_A, subA, rows * n, MPI_INT, &status);

    MPI_File_read_at_all(fh_B, offset_B, B, n * n, MPI_INT, &status);

    MPI_File_close(&fh_A);
    MPI_File_close(&fh_B);

    cudaMatrixMultiply(subA, B, subC, n, size, rank);

    if (rank == 0) {
        int *C;
        cudaMallocManaged(&C, n * n * sizeof(int));
        MPI_Gather(subC, rows * n, MPI_INT, C, rows * n, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_File_open(MPI_COMM_SELF, "matrix_C.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_C);
        MPI_File_write(fh_C, C, n * n, MPI_INT, &status);
        MPI_File_close(&fh_C);

        cudaFree(C);
    } else {
        MPI_Gather(subC, rows * n, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    cudaFree(subA);
    cudaFree(B);
    cudaFree(subC);

    MPI_Finalize();
    return 0;
}

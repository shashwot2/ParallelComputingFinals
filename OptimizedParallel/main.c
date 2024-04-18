#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix_operations.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    performMatrixMultiplicationCUDA(A, B, C, n);

    MPI_Finalize();
    return 0;
}

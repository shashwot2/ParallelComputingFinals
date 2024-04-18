#include <stdio.h>
#include <stdlib.h>

void add(int **A, int **B, int **result, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            result[i][j] = A[i][j] + B[i][j];
}

void subtract(int **A, int **B, int **result, int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            result[i][j] = A[i][j] - B[i][j];
}

void strassen(int **A, int **B, int **C, int n)
{
    if (n == 1)
    {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    int new_size = n / 2;
    int **A11, **A12, **A21, **A22;
    int **B11, **B12, **B21, **B22;
    int **C11, **C12, **C21, **C22;
    int **M1, **M2, **M3, **M4, **M5, **M6, **M7;
    int **temp1, **temp2;

    // Dynamic memory allocation for all submatrices and temporary matrices
    A11 = (int **)malloc(new_size * sizeof(int *));
    A12 = (int **)malloc(new_size * sizeof(int *));
    A21 = (int **)malloc(new_size * sizeof(int *));
    A22 = (int **)malloc(new_size * sizeof(int *));
    B11 = (int **)malloc(new_size * sizeof(int *));
    B12 = (int **)malloc(new_size * sizeof(int *));
    B21 = (int **)malloc(new_size * sizeof(int *));
    B22 = (int **)malloc(new_size * sizeof(int *));
    C11 = (int **)malloc(new_size * sizeof(int *));
    C12 = (int **)malloc(new_size * sizeof(int *));
    C21 = (int **)malloc(new_size * sizeof(int *));
    C22 = (int **)malloc(new_size * sizeof(int *));
    M1 = (int **)malloc(new_size * sizeof(int *));
    M2 = (int **)malloc(new_size * sizeof(int *));
    M3 = (int **)malloc(new_size * sizeof(int *));
    M4 = (int **)malloc(new_size * sizeof(int *));
    M5 = (int **)malloc(new_size * sizeof(int *));
    M6 = (int **)malloc(new_size * sizeof(int *));
    M7 = (int **)malloc(new_size * sizeof(int *));
    temp1 = (int **)malloc(new_size * sizeof(int *));
    temp2 = (int **)malloc(new_size * sizeof(int *));
    for (int i = 0; i < new_size; i++)
    {
        A11[i] = (int *)malloc(new_size * sizeof(int));
        A12[i] = (int *)malloc(new_size * sizeof(int));
        A21[i] = (int *)malloc(new_size * sizeof(int));
        A22[i] = (int *)malloc(new_size * sizeof(int));
        B11[i] = (int *)malloc(new_size * sizeof(int));
        B12[i] = (int *)malloc(new_size * sizeof(int));
        B21[i] = (int *)malloc(new_size * sizeof(int));
        B22[i] = (int *)malloc(new_size * sizeof(int));
        C11[i] = (int *)malloc(new_size * sizeof(int));
        C12[i] = (int *)malloc(new_size * sizeof(int));
        C21[i] = (int *)malloc(new_size * sizeof(int));
        C22[i] = (int *)malloc(new_size * sizeof(int));
        M1[i] = (int *)malloc(new_size * sizeof(int));
        M2[i] = (int *)malloc(new_size * sizeof(int));
        M3[i] = (int *)malloc(new_size * sizeof(int));
        M4[i] = (int *)malloc(new_size * sizeof(int));
        M5[i] = (int *)malloc(new_size * sizeof(int));
        M6[i] = (int *)malloc(new_size * sizeof(int));
        M7[i] = (int *)malloc(new_size * sizeof(int));
        temp1[i] = (int *)malloc(new_size * sizeof(int));
        temp2[i] = (int *)malloc(new_size * sizeof(int));
    }

    // Subdivide input matrices A and B into submatrices A11, A12, A21, A22, B11, B12, B21, B22
    for (int i = 0; i < new_size; i++)
    {
        for (int j = 0; j < new_size; j++)
        {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + new_size];
            A21[i][j] = A[i + new_size][j];
            A22[i][j] = A[i + new_size][j + new_size];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + new_size];
            B21[i][j] = B[i + new_size][j];
            B22[i][j] = B[i + new_size][j + new_size];
        }
    }

    // Calculate M1 = (A11 + A22) * (B11 + B22)
    add(A11, A22, temp1, new_size);
    add(B11, B22, temp2, new_size);
    strassen(temp1, temp2, M1, new_size);

    // Calculate M2 = (A21 + A22) * B11
    add(A21, A22, temp1, new_size);
    strassen(temp1, B11, M2, new_size);

    // Calculate M3 = A11 * (B12 - B22)
    subtract(B12, B22, temp2, new_size);
    strassen(A11, temp2, M3, new_size);

    // Calculate M4 = A22 * (B21 - B11)
    subtract(B21, B11, temp2, new_size);
    strassen(A22, temp2, M4, new_size);

    // Calculate M5 = (A11 + A12) * B22
    add(A11, A12, temp1, new_size);
    strassen(temp1, B22, M5, new_size);

    // Calculate M6 = (A21 - A11) * (B11 + B12)
    subtract(A21, A11, temp1, new_size);
    add(B11, B12, temp2, new_size);
    strassen(temp1, temp2, M6, new_size);

    // Calculate M7 = (A12 - A22) * (B21 + B22)
    subtract(A12, A22, temp1, new_size);
    add(B21, B22, temp2, new_size);
    strassen(temp1, temp2, M7, new_size);

    // Calculate C11 = M1 + M4 - M5 + M7
    add(M1, M4, temp1, new_size);
    subtract(temp1, M5, temp2, new_size);
    add(temp2, M7, C11, new_size);

    // Calculate C12 = M3 + M5
    add(M3, M5, C12, new_size);

    // Calculate C21 = M2 + M4
    add(M2, M4, C21, new_size);

    // Calculate C22 = M1 - M2 + M3 + M6
    subtract(M1, M2, temp1, new_size);
    add(temp1, M3, temp2, new_size);
    add(temp2, M6, C22, new_size);

    // Group results back into matrix C
    for (int i = 0; i < new_size; i++)
    {
        for (int j = 0; j < new_size; j++)
        {
            C[i][j] = C11[i][j];
            C[i][j + new_size] = C12[i][j];
            C[i + new_size][j] = C21[i][j];
            C[i + new_size][j + new_size] = C22[i][j];
        }
    }

    // Free all allocated memory
    for (int i = 0; i < new_size; i++)
    {
        free(A11[i]);
        free(A12[i]);
        free(A21[i]);
        free(A22[i]);
        free(B11[i]);
        free(B12[i]);
        free(B21[i]);
        free(B22[i]);
        free(C11[i]);
        free(C12[i]);
        free(C21[i]);
        free(C22[i]);
        free(M1[i]);
        free(M2[i]);
        free(M3[i]);
        free(M4[i]);
        free(M5[i]);
        free(M6[i]);
        free(M7[i]);
        free(temp1[i]);
        free(temp2[i]);
    }
    free(A11);
    free(A12);
    free(A21);
    free(A22);
    free(B11);
    free(B12);
    free(B21);
    free(B22);
    free(C11);
    free(C12);
    free(C21);
    free(C22);
    free(M1);
    free(M2);
    free(M3);
    free(M4);
    free(M5);
    free(M6);
    free(M7);
    free(temp1);
    free(temp2);
}

int isPowerOfTwo(int x) {
    return (x > 0) && ((x & (x - 1)) == 0);
}

int main(int argc, char *argv[])
{
     if (argc < 2) {
        printf("Usage: %s <matrix size>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);

    if (!isPowerOfTwo(n)) {
        printf("Error: The matrix size must be a power of 2.\n");
        return 1;
    }
    int **A, **B, **C;

    // Allocate memory for matrices A, B, and C
    A = (int **)malloc(n * sizeof(int *));
    B = (int **)malloc(n * sizeof(int *));
    C = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
    {
        A[i] = (int *)malloc(n * sizeof(int));
        B[i] = (int *)malloc(n * sizeof(int));
        C[i] = (int *)malloc(n * sizeof(int));
    }

    // Initialize matrices A and B with some values
    printf("Matrix A:\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i][j] = i + j; 
            B[i][j] = i - j; 
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }
    printf("Matrix B:\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d ", B[i][j]);
        }
        printf("\n");
    }

    // Perform matrix multiplication using Strassen's algorithm
    strassen(A, B, C, n);

    // Print the result matrix C
    printf("Matrix C (Result):\n");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    // Free all allocated memory
    for (int i = 0; i < n; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}

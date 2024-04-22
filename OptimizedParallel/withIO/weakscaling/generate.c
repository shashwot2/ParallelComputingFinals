#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <matrix_size> <output_file_a> <output_file_b>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    char *output_file_a = argv[2];
    char *output_file_b = argv[3];

    // Seed the random number generator
    srand(time(NULL));

    // Allocate memory for the matrices
    int **a = (int **)malloc(N * sizeof(int *));
    int **b = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        a[i] = (int *)malloc(N * sizeof(int));
        b[i] = (int *)malloc(N * sizeof(int));
    }

    // Generate random numbers for both matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = rand() % 10;
            b[i][j] = rand() % 10;
        }
    }

    // Write matrices to files
    FILE *fa = fopen(output_file_a, "wb");
    FILE *fb = fopen(output_file_b, "wb");
    if (!fa || !fb) {
        printf("Error opening files!\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        fwrite(a[i], sizeof(int), N, fa);
        fwrite(b[i], sizeof(int), N, fb);
    }

    // Close files
    fclose(fa);
    fclose(fb);

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(a[i]);
        free(b[i]);
    }
    free(a);
    free(b);

    return 0;
}

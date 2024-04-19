#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

void generateAndWriteMatrix(const char* filename, int N) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open the file for writing." << std::endl;
        exit(1);
    }

    srand((unsigned)time(0));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int value = rand() % 100; 
            file.write(reinterpret_cast<const char*>(&value), sizeof(int));
        }
    }

    file.close();
}

int main() {
    int N = 4;  
    generateAndWriteMatrix("matrix_A.bin", N);
    generateAndWriteMatrix("matrix_B.bin", N);

    return 0;
}

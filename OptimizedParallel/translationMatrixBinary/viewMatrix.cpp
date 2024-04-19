#include <iostream>
#include <fstream>
#include <vector>
#include <cstring> // For strerror
#include <cerrno>  // For errno

void readAndPrintMatrix(const char* filename, int N) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open the file for reading: " << filename << " - " << strerror(errno) << std::endl;
        return;
    }

    std::vector<int> matrix(N * N);
    file.read(reinterpret_cast<char*>(matrix.data()), N * N * sizeof(int));
    if (!file.good()) {
        std::cerr << "Error or incomplete read from the file. Read bytes: " << file.gcount() << ", expected: " << N * N * sizeof(int) << std::endl;
        return;
    }

    std::cout << "Matrix from " << filename << ":" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    file.close();
}

int main() {
    int N = 4; 

    readAndPrintMatrix("../matrix_A.bin", N);
    readAndPrintMatrix("../matrix_B.bin", N);
    readAndPrintMatrix("../matrix_C.bin", N);

    return 0;
}

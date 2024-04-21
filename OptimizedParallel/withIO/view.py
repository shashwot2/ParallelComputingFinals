import numpy as np

N = 4  
filenameinput1 = 'matrix_a.dat'
filenameinput2 = 'matrix_b.dat'
filename = 'result_matrix.dat'
input1matrix = np.fromfile(filenameinput1, dtype=np.int32).reshape(N, N)
input2matrix = np.fromfile(filenameinput2, dtype=np.int32).reshape(N, N)
resultmatrix = np.fromfile(filename, dtype=np.int32).reshape(N, N)

print("Matrix A:")
print(input1matrix)
print("Matrix B:")
print(input2matrix)

print("Matrix C:")
print(resultmatrix)
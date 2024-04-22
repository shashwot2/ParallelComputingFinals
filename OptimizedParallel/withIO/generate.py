import sys
import numpy as np

if len(sys.argv) != 4:
    print("Usage: python generate_matrices.py <matrix_size> <output_file_a> <output_file_b>")
    sys.exit(1)

N = int(sys.argv[1])
output_file_a = sys.argv[2]
output_file_b = sys.argv[3]

a = np.random.randint(0, 10, size=(N, N))
b = np.random.randint(0, 10, size=(N, N))

a.astype('int32').tofile(output_file_a)
b.astype('int32').tofile(output_file_b)
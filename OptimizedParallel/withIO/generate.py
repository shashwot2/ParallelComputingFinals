import numpy as np

N = 4  
a = np.random.randint(0, 10, size=(N, N))
b = np.random.randint(0, 10, size=(N, N))

a.astype('int32').tofile('matrix_a.dat')
b.astype('int32').tofile('matrix_b.dat')

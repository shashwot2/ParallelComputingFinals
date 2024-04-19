CC=mpicc
NVCC=nvcc
CUDA_PATH=/usr/local/cuda/lib64

OBJS=main.o matrix_operations.o

CFLAGS=-I/usr/local/cuda/include
LDFLAGS=-L$(CUDA_PATH) -lcudart

EXEC=program

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

main.o: main.c
	$(CC) -c $< $(CFLAGS)

matrix_operations.o: matrix_operations.cu
	$(NVCC) -c $< $(CFLAGS)

clean:
	rm -f $(OBJS) $(EXEC)

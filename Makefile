toyNet : matrix.o nn.o main.o
	clang matrix.o nn.o main.o -lm -o toyNet

matrix.o : matrix.c
	clang -c matrix.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -o matrix.o

nn.o : nn.c
	clang -c nn.c -O3 -o nn.o

main.o : main.c
	clang -c main.c -O3 -o main.o

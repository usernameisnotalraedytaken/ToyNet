toyNet : matrix.o nn.o main.o train.o test.o
	clang matrix.o nn.o main.o -lm -o toyNet_demo
	clang matrix.o nn.o train.o -lm -o toyNet_train
	clang matrix.o nn.o test.o -lm -o toyNet_test

matrix.o : matrix.c
	clang -c matrix.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -o matrix.o

nn.o : nn.c
	clang -c nn.c -O3 -o nn.o

train.o : train.c
	clang -c train.c -O3 -o train.o

test.o : test.c
	clang -c test.c -O3 -o test.o

main.o : main.c
	clang -c main.c -O3 -o main.o

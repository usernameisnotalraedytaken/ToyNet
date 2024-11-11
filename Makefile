toyNet : matrix.o nn.o main.o train.o test.o
	clang matrix.o nn.o main.o -lm -o toyNet_demo.exe
	clang matrix.o nn.o train.o -lm -o toyNet_train.exe
	clang matrix.o nn.o test.o -lm -o toyNet_test.exe

matrix.o : matrix.c
	clang -c matrix.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -O3 -mavx2 -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o matrix.o

nn.o : nn.c
	clang -c nn.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -O3 -mavx2 -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o nn.o

train.o : train.c
	clang -c train.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -O3 -mavx2 -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o train.o

test.o : test.c
	clang -c test.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -O3 -mavx2 -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o test.o

main.o : main.c
	clang -c main.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -O3 -mavx2 -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o main.o

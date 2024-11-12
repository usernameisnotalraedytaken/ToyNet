toyNet : matrix.o nn.o main.o train.o test.o
	gcc matrix.o nn.o main.o -lm -o toyNet_demo.exe
	gcc matrix.o nn.o train.o -lm -o toyNet_train.exe
	gcc matrix.o nn.o test.o -lm -o toyNet_test.exe

matrix.o : matrix.c
	gcc -c matrix.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o matrix.o

nn.o : nn.c
	gcc -c nn.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o nn.o

train.o : train.c
	gcc -c train.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o train.o

test.o : test.c
	gcc -c test.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o test.o

main.o : main.c
	gcc -c main.c -O3 -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o main.o

toyNet : matrix.o nn.o main.o train.o test.o
	gcc matrix.o nn.o main.o -lm -o toyNet_demo.exe
	gcc matrix.o nn.o train.o -lm -o toyNet_train.exe
	gcc matrix.o nn.o test.o -lm -o toyNet_test.exe
	g++ benchmark.cpp -o benchmark.exe

.PHONY: clean

matrix.o : matrix.c
	gcc -c matrix.c -Ofast -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -ffast-math -o matrix.o

nn.o : nn.c
	gcc -std=c2x -c nn.c -Ofast -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -lpthread -o nn.o

train.o : train.c
	gcc -c train.c -Ofast -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o train.o

test.o : test.c
	gcc -c test.c -Ofast -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o test.o

main.o : main.c
	gcc -c main.c -Ofast -mavx2 -march=native -fexcess-precision=fast -ftree-vectorize -ftree-slp-vectorize -ftree-loop-if-convert -fvect-cost-model=dynamic -fsimd-cost-model=dynamic -o main.o

clean : 
	rm matrix.o nn.o train.o test.o main.o

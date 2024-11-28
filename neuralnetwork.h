#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LOG(...) fprintf(stderr, "%s: Ln %d: %s\n", __FILE__, __LINE__, __VA_ARGS__)

#define M_PI 3.14159265358979323846
typedef float real;
typedef struct
{
    int row_size, col_size;
    real *data;
} Matrix;

typedef struct
{
    int innodes, hidenodes, outnodes;
    real learnrate;
    Matrix Weight_in_to_hidden, Weight_hidden_to_out;
} NeuralNetwork;

typedef struct
{
    Matrix *node;
    Matrix error;
    Matrix lastoutput;
    Matrix nextoutput;
    real learnrate;
} ErrorFeedbackCorrectionArgs;


// Matrix algorithms.

void Fill(Matrix *mat, int rows, int cols, real value);
void Print(Matrix *mat);
void ReLU(Matrix *m);
void Sigmoid(Matrix *m);
void freeMatrix(Matrix *mat);
Matrix Id(int size);
void valMul(real val, Matrix *m);
real normalDistributionRandom();
real xavierNormalInit(int fanIn, int fanOut);
void Normalize(Matrix *m);
Matrix XavierRand(int row, int col);
void Add(Matrix a, Matrix b, Matrix *c);
void Minus(Matrix a, Matrix b, Matrix *c);
void Cross(Matrix a, Matrix b, Matrix *c);
void Mul(Matrix a, Matrix b, Matrix *c);
Matrix Tr(Matrix a);
void ErrorFeedbackCorrection(Matrix *node, Matrix error, Matrix lastoutput, Matrix nextoutput, real learnrate);
void ErrorFeedbackCorrectionThread(void *p);
Matrix addSpice(Matrix v, real rate);

// NeuralNetwork algorithms.

NeuralNetwork InitNN(int innodes, int hidenodes, int outnodes, real learnrate);
void TrainNN(NeuralNetwork *nn, Matrix input, Matrix target, int n);
void QueryNN(NeuralNetwork *nn, Matrix input, Matrix *output);
void freeNN(NeuralNetwork *nn);
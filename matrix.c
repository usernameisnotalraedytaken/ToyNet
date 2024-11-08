#include "neuralnetwork.h"
//Matrix Initializers and Modifiers.
void Fill(Matrix *mat, int rows, int cols, real value)
{
    if (mat->col_size == cols && mat->row_size == rows)
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                mat->data[i][j] = value;
        return;
    }
    if (mat->col_size != 0 || mat->row_size != 0)
    {
        printf("!Error: trying to fill a non-empty matrix with new size.\n");
        printf("Original matrix size: %d x %d\n", mat->row_size, mat->col_size);
        printf("New matrix size: %d x %d\n", rows, cols);
        mat->col_size = cols;
        mat->row_size = rows;
        exit(EXIT_FAILURE);
    }
    mat->row_size = rows;
    mat->col_size = cols;
    mat->data = (real **)malloc(rows * sizeof(real *));
    if (mat->data == NULL)
    {
        fprintf(stderr, "Memory allocation failed for matrix data.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        mat->data[i] = (real *)malloc(cols * sizeof(real));
        if (mat->data[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for row %d.\n", i);
            for (int j = 0; j < i; j++)
                free(mat->data[j]);
            free(mat->data);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < cols; j++)
            mat->data[i][j] = value;
    }
}

void Print(Matrix *mat)
{
    if (mat->col_size == 0 || mat->row_size == 0)
    {
        printf("!Error: trying to print a matrix with zero size.\n");
        exit(EXIT_FAILURE);
    }
    puts("----------------------------------------");
    for (int i = 0; i < mat->row_size; i++)
    {
        for (int j = 0; j < mat->col_size; j++)
            printf("%10.5f ", mat->data[i][j]);
        printf("\n");
    }
    puts("----------------------------------------\n");
}

void ReLU(Matrix *m)
{
    for (int i = 0; i < m->row_size; i++)
        for (int j = 0; j < m->col_size; j++)
            if(m->data[i][j] < 0)
                m->data[i][j] = 0;
}

void Sigmoid(Matrix *m)
{
    int row = m->row_size;
    int col = m->col_size;
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            m->data[i][j] = 1 / (1 + exp(-m->data[i][j]));
}

void freeMatrix(Matrix *mat)
{
    if (mat == NULL || mat->data == NULL)
    {
        printf("!Error: trying to free a NULL matrix.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < mat->row_size; i++)
    {
        free(mat->data[i]);
        mat->data[i] = NULL;
    }
    free(mat->data);
    mat->data = NULL;
}

//Matrix Generators.
Matrix Id(int size)
{
    Matrix m = {0, 0, NULL};
    Fill(&m, size, size, 0);
    for (int i = 0; i < size; i++)
        m.data[i][i] = 1;
    return m;
}

void valMul(real val, Matrix *m)
{
    if (m->col_size == 0 || m->row_size == 0)
    {
        printf("!Error: trying to multiply a matrix with zero size.\n");
        exit(EXIT_FAILURE);
    }
    int row = m->row_size;
    int col = m->col_size;
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            m->data[i][j] *= val;
    //return m;
}
real normalDistributionRandom()
{
    real u1 = (real)rand() / RAND_MAX;
    real u2 = (real)rand() / RAND_MAX;
    real r = sqrt(-2.0 * log(u1));
    real theta = 2.0 * M_PI * u2;
    return r * cos(theta);
}

real xavierNormalInit(int fanIn, int fanOut)
{
    real stdDev = sqrt(2.0 / (fanIn + fanOut));
    return normalDistributionRandom() * stdDev;
}

void Normalize(Matrix *m)
{
    real Min = 0, Max = 0;
    int row = m->row_size;
    int col = m->col_size;
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
        {
            if(m->data[i][j] > Max)Max = m->data[i][j];
            if(m->data[i][j] < Min)Min = m->data[i][j];
        }
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            m->data[i][j] = (m->data[i][j] - Min) / (Max - Min) * 0.99 + 0.01;
}
Matrix XavierRand(int row, int col)
{
    Matrix m = {0, 0, NULL};
    Fill(&m, row, col, 0);
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            m.data[i][j] = xavierNormalInit(row, col);
    return m;
}
//Add two Matrixes(A + B) directly.
void Add(Matrix a, Matrix b, Matrix *c)
{
    if((a.row_size != b.row_size) || (a.col_size != b.col_size))
    {
        printf("!Error: trying to add two matrices with different sizes.\n");
        exit(EXIT_FAILURE);
    }
    //Fill(c, a.row_size, a.col_size, 0);
    int row = a.row_size;
    int col = a.col_size;
    for (int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            c->data[i][j] = a.data[i][j] + b.data[i][j];
}

//Calculate Matrix a - b directly.
void Minus(Matrix a, Matrix b, Matrix *c)
{
    if((a.row_size != b.row_size) || (a.col_size != b.col_size))
    {
        printf("!Error: trying to minus two matrices with different sizes.\n");
        exit(EXIT_FAILURE);
    }
    Fill(c, a.row_size, a.col_size, 0);
    int row = a.row_size;
    int col = a.col_size;
    for (int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            c->data[i][j] = a.data[i][j] - b.data[i][j];
}
//Multiplication through a[i][j] * b[i][j]
void Cross(Matrix a, Matrix b, Matrix *c)
{
    if((a.row_size != b.row_size) || (a.col_size != b.col_size))
    {
        printf("!Error: trying to cross two matrices with different sizes.\n");
        exit(EXIT_FAILURE);
    }
    Fill(c, a.row_size, a.col_size, 0);
    int row = a.row_size;
    int col = a.col_size;
    for (int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            c->data[i][j] = a.data[i][j] * b.data[i][j];
}

void Mul(Matrix a, Matrix b, Matrix *c)
{
    int m = a.row_size;
    int n = a.col_size;
    int p = b.col_size;
    if(n != b.row_size)
    {
        printf("!Error: trying to multiply two matrices with not matching sizes.\n");
        exit(EXIT_FAILURE);
    }
    Fill(c, m, p, 0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
        {
            real val = 0;
            for (int k = 0; k < n; k++)
                val += a.data[i][k] * b.data[k][j];
            c->data[i][j] = val;
        }
}

Matrix Tr(Matrix a)
{
    Matrix b = {0, 0, NULL};
    Fill(&b, a.col_size, a.row_size, 0);
    for (int i = 0; i < a.col_size; i++)
        for (int j = 0; j < a.row_size; j++)
            b.data[i][j] = a.data[j][i];
    return b;
}

void ErrorFeedbackCorrection(Matrix *node, Matrix error, Matrix lastoutput, Matrix nextoutput, real learnrate)
{
    int m = lastoutput.row_size;
    int n = lastoutput.col_size;
    Matrix Ones = {0, 0, NULL}, Cro = {0, 0, NULL}, Res = {0, 0, NULL};
    Fill(&Ones, m, n, 1.0);
    Fill(&Cro, m, n, 0);
    Fill(&Res, m, n, 0);
    Minus(Ones, lastoutput, &Res);
    Cross(error, lastoutput, &Cro);
    Matrix Res3 = {0, 0, NULL};
    Cross(Res, Cro, &Res3);
    Matrix Res5 = {0, 0, NULL};
    Res5 = Tr(nextoutput);
    Matrix Res2 = {0, 0, NULL};
    Mul(Res3, Res5, &Res2);
    valMul(learnrate, &Res2);
    Add(*node, Res2, node);
    freeMatrix(&Ones);
    freeMatrix(&Cro);
    freeMatrix(&Res);
    freeMatrix(&Res2);
    freeMatrix(&Res3);
    freeMatrix(&Res5);
}

Matrix addSpice(Matrix v, real rate)
{
    Matrix r = {0, 0, NULL};
    Fill(&r, v.row_size, v.col_size, 0);
    for(int i = 0; i < v.row_size; ++i)
        for(int j = 0; j < v.col_size; ++j)
            r.data[i][j] = v.data[i][j] + rate * normalDistributionRandom();
    return r;
}

#include <stdio.h>
#include <time.h>
#include "neuralnetwork.h"

int read(Matrix *vec)
{
    int n = 0;
    char c1 = getchar();
    while(c1>'9'||c1<'0')c1=getchar();
    while(c1>='0'&&c1<='9')n=n*10+c1-'0',c1=getchar();
    vec->row_size = 1;
    vec->col_size = 28*28;
    for(unsigned i = 0; i < 28*28; ++i)
    {
        unsigned char u = 0;
        char c = getchar();
        while(c>'9'||c<'0')c=getchar();
        while(c>='0'&&c<='9')u=u*10+c-'0',c=getchar();
        vec->data[i] = u;
    }
    return n;
}

void InputWeights(NeuralNetwork *nn)
{
    freopen("weights1.txt", "r", stdin);
    scanf("%d%d", &nn->Weight_in_to_hidden.row_size, &nn->Weight_in_to_hidden.col_size);
    int rows = nn->Weight_in_to_hidden.row_size;
    int cols = nn->Weight_in_to_hidden.col_size;
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            scanf("%f", &(nn->Weight_in_to_hidden.data[i * cols + j]));
    freopen("weights2.txt", "r", stdin);
    scanf("%d%d", &nn->Weight_hidden_to_out.row_size, &nn->Weight_hidden_to_out.col_size);
    rows = nn->Weight_hidden_to_out.row_size;
    cols = nn->Weight_hidden_to_out.col_size;
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            scanf("%f", &(nn->Weight_hidden_to_out.data[i * cols + j]));
    printf("Weights read from files.\n");
}

void QueryOneItem(NeuralNetwork *nn)
{
    int correct = 0;
    Matrix v = {0, 0, NULL}, w = {0, 0, NULL};
    Fill(&v, 1, 784, 0);
    Fill(&w, 10, 1, 0);
    printf("Start to query one item:\n");
    freopen("data.csv", "r", stdin);
    int target = read(&v);
    Normalize(&v);
    QueryNN(nn, v, &w);
    Normalize(&w);
    int max_index = 0;
    printf("Possibilities:\n");
    for(int j = 0; j < 10; ++j)
        if(w.data[j * w.col_size] >= w.data[max_index * w.col_size])
            max_index = j;
    for(int j = 0; j < 10; ++j)
    {
        if(j == max_index)
            printf("\033[1;92m%d     \033[0m", j);
        else
            printf("%d     ", j);
    }
    puts("");
    for(int j = 0; j < 10; ++j)
    {
        if(j == max_index)
            printf("\033[1;92m%.03f \033[0m", w.data[j * w.col_size]);
        else
            printf("%.03f ", w.data[j * w.col_size]);
    }
    freeMatrix(&v);
    freeMatrix(&w);
    printf("\nPrediction = \033[1;92m%d\033[0m\n", max_index);
}

int main()
{
    clock_t start, end;
    start = clock();
    printf("MNIST MLP Classifier Test Session, optimized version.\n");
    srand(time(NULL));
    NeuralNetwork nn = InitNN(784, 200, 10, 0.1);
    printf("`nn` initialized.\n");
    InputWeights(&nn);
    QueryOneItem(&nn);
    end = clock();
    double elapsed_secs = (end - start) * 1.0 / CLOCKS_PER_SEC;
    printf("Elapsed time: %f seconds.\n", elapsed_secs);   
    return 0;
}
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
        vec->data[0][i] = u;
    }
    return n;
}

void InputWeights(NeuralNetwork *nn)
{
    freopen("weights1.txt", "r", stdin);
    int rows = nn->Weight_in_to_hidden.row_size;
    int cols = nn->Weight_in_to_hidden.col_size;
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            scanf("%f", &(nn->Weight_in_to_hidden.data[i][j]));
    freopen("weights2.txt", "r", stdin);
    rows = nn->Weight_hidden_to_out.row_size;
    cols = nn->Weight_hidden_to_out.col_size;
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            scanf("%f", &(nn->Weight_hidden_to_out.data[i][j]));
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
        printf("%d     ", j);
    puts("");
    for(int j = 0; j < 10; ++j)
    {
        printf("%.03f ", w.data[j][0]);
        if(w.data[j][0] >= w.data[max_index][0])
            max_index = j;
    }
    freeMatrix(&v);
    freeMatrix(&w);
    printf("\nPrediction = %d\n", max_index);
}

int main()
{
    clock_t start, end;
    start = clock();
    printf("MNIST MLP Classifier Test Session, optimized version.\n");
    srand(time(NULL));
    NeuralNetwork nn = InitNN(784, 256, 10, 0.1);
    printf("`nn` initialized.\n");
    InputWeights(&nn);
    QueryOneItem(&nn);
    end = clock();
    double elapsed_secs = (end - start) * 1.0 / CLOCKS_PER_SEC;
    printf("Elapsed time: %f seconds.\n", elapsed_secs);   
    return 0;
}
